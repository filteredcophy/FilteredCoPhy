import torch
import argparse
from Dataloader.Video_Loaders import blocktowerCF_Video, ballsCF_Video, collisionCF_Video
import torch.nn as nn
from Models.CoDy import CoDy
from Models.Derendering import Derendering
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
import pickle
import os
import numpy as np
from misc.MOT_Metrics import MOTAccumulator
from misc.background_substraction import remove_background

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="balls", type=str,
                    help="Datasets, should be one of 'blocktower', 'balls' or 'collision")
parser.add_argument('--n_keypoints', default=4, type=int, help="Number of keypoints")
parser.add_argument('--n_coefficients', default=4, type=int, help="Number of coefficients")
parser.add_argument('--mode', default="fixed", type=str,
                    help="'fixed': use fixed dilatation filter bank. 'learned': learn the filters via gradient descent")
parser.add_argument('--derendering_path', default='', type=str, help="Path to the weights of the de-rendering module")
parser.add_argument('--cody_path', default='', type=str, help="Path to the weights of CoDy")
parser.add_argument('--output_path', default='', type=str, help="Output file")
args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

MSE = nn.MSELoss()

datasets = {"blocktower": blocktowerCF_Video, "balls": ballsCF_Video, "collision": collisionCF_Video}


class Pipeline(nn.Module):
    def __init__(self):
        super(Pipeline, self).__init__()
        self.derendering = Derendering(n_keypoints=args.n_keypoints,
                                       mode=args.mode,
                                       n_coefficients=args.n_coefficients,
                                       device=device).to(device)

        self.cody = CoDy(state_size=14, cf_size=32, emb_size=128,
                         n_layers=2, n_gn=1, z_size=256, hidden_size=64).to(device)

    def load(self):
        self.derendering.load_state_dict(torch.load(args.derendering_path, map_location=device))
        self.cody.load_state_dict(torch.load(args.cody_path, map_location=device))

    def forward(self, rgb_ab, rgb_cd):
        # Compute ground truth keypoints
        states_ab, _ = self.get_keypoints(rgb_ab)
        states_ab = self.add_speed(states_ab)

        states_cd, features_cd = self.get_keypoints(rgb_cd)
        states_cd = self.add_speed(states_cd)

        # Forecast D
        c = states_cd[:, 0]
        state_hat = self.cody(states_ab, c, horizon=rgb_ab.shape[1] - 1)

        # Extract keypoints and coef from prediction
        predicted_keypoints = state_hat[..., :2].view(-1, args.n_keypoints, 2)
        predicted_coefficients = state_hat[..., 2:3 + args.n_coefficients].view(-1, args.n_keypoints,
                                                                                args.n_coefficients + 1)
        gt_keypoints = states_cd[..., :2].view(-1, args.n_keypoints, 2)
        gt_coefficients = states_cd[..., 2:3 + args.n_coefficients].view(-1, args.n_keypoints, args.n_coefficients + 1)

        # Duplicate the features T times
        features_cd = features_cd.view(-1, rgb_ab.shape[1], 16, 28, 28)
        first_features = features_cd[:, :1].repeat(1, rgb_ab.shape[1], 1, 1, 1)
        first_features = first_features.view(-1, 16, 28, 28)

        predicted_reconstruction = self.derendering.create_img(first_features, predicted_keypoints,
                                                               predicted_coefficients)
        gt_reconstruction = self.derendering.create_img(first_features, gt_keypoints, gt_coefficients)
        B, T, C, H, W = rgb_cd.shape
        return states_cd, state_hat, predicted_reconstruction.view(B, -1, C, H, W), gt_reconstruction.view(B, -1, C, H,
                                                                                                           W)

    def add_speed(self, x):
        B, T, K, S = x.shape
        speed = torch.cat([torch.zeros(B, 1, K, S).to(x.device), x[:, 1:] - x[:, :-1]], dim=1)
        x = torch.cat([x, speed], dim=-1)
        return x

    def get_keypoints(self, rgb):
        B, T, C, H, W = rgb.shape
        rgb = rgb.view(-1, C, H, W)
        features, heatmap, coef = self.derendering.encoder(rgb)
        parameter = self.derendering.extractor(torch.flatten(coef, start_dim=2))
        parameter = torch.sigmoid(parameter)
        keypoints = self.derendering.get_keypoint_location(heatmap)

        parameter = parameter.view(B, T, -1, args.n_coefficients + 1)
        keypoints = keypoints.view(B, T, -1, 2)

        states = torch.cat([keypoints, parameter], dim=-1)
        return states, features


def evaluate():
    print(args)
    dataloader = DataLoader(
        datasets[args.dataset](mode='test', resolution=112, sampling_mode='full', load_ab=True, load_state=True),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    model = Pipeline().to(device)
    model.load()

    model.eval()
    logs = {}

    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            acc = MOTAccumulator()
            rgb_ab = x["rgb_ab"].to(device)
            rgb_cd = x["rgb_cd"].to(device)

            keypoints_gt, keypoints_hat, rgb_hat, gt_rgb = model(rgb_ab, rgb_cd)

            cd_img = rgb_cd.cpu().numpy()[0].transpose(0, 2, 3, 1)
            predictions = rgb_hat.cpu().numpy()[0].transpose(0, 2, 3, 1)
            gt_img = gt_rgb.cpu().numpy()[0].transpose(0, 2, 3, 1)

            kpts_loss = ((keypoints_gt[..., :2] - keypoints_hat[..., :2]) ** 2).mean().detach().cpu().numpy()

            states = x["pose_2D_cd"].float()
            stability = ((states[:, 1:] - states[:, :-1]) ** 2).sum([0, 1, -1])

            keypoints_hat[..., :2] = ((keypoints_hat[..., :2] + 1) / 2) * 112
            keypoints_hat[..., :2] = torch.stack([keypoints_hat[..., 1], keypoints_hat[..., 0]], dim=-1)

            keypoints_hat = keypoints_hat.detach().cpu().numpy()[0]
            states = states.detach().cpu().numpy()[0]

            background_mask = remove_background(cd_img, args.dataset)

            PSNR, MK_PSNR, GT_PSNR = [], [], []
            for t in range(keypoints_gt.shape[1]):
                PSNR.append(peak_signal_noise_ratio(cd_img[t], predictions[t] + 1e-4, data_range=2))
                GT_PSNR.append(peak_signal_noise_ratio(cd_img[t], gt_img[t] + 1e-4, data_range=2))
                MK_PSNR.append(peak_signal_noise_ratio(cd_img[t] * background_mask[t],
                                                       predictions[t] * background_mask[t] + 1e-4, data_range=2))

                activities = keypoints_hat[t, :, 6]
                acc.update(states[t, :, :2], keypoints_hat[t, :, :2], activities)

            motp, mota = acc.compute_metrics()
            logs[i] = {'PSNR': PSNR, 'KPTS_loss': kpts_loss, 'MASKED_PSNR': MK_PSNR,
                       'MOTA': mota, "MOTP": motp, 'stability': stability}

        pickle.dump(logs, open(args.output_path, 'wb'))


if __name__ == '__main__':
    evaluate()
