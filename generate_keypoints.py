import torch
import argparse
from Dataloader.Video_Loaders import ballsCF_Video, blocktowerCF_Video, collisionCF_Video
from Models.Derendering import Derendering
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="blocktower", type=str,
                    help="Datasets, should be one of 'blocktower', 'balls' or 'collision")
parser.add_argument('--n_coefficients', default=4, type=int, help="Number of coefficients")
parser.add_argument('--n_keypoints', default=4, type=int, help="Number of keypoints")
parser.add_argument('--mode', default="fixed", type=str,
                    help="'fixed': use fixed dilatation filter bank. 'learned': learn the filters via gradient descent")
parser.add_argument('--model_weight_path', default='', type=str, help="Path to the weights of the de-rendering model")
parser.add_argument('--output_path', default='', type=str, help="Path where to store the keypoints")
args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
datasets = {"blocktower": blocktowerCF_Video, "balls": ballsCF_Video, "collision": collisionCF_Video}


def main():
    train_dataloader = DataLoader(
        datasets[args.dataset](mode='train', resolution=112, sampling_mode='full', load_ab=True),
        batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
    val_dataloader = DataLoader(
        datasets[args.dataset](mode='val', resolution=112, sampling_mode='full', load_ab=True),
        batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
    test_dataloader = DataLoader(
        datasets[args.dataset](mode='test', resolution=112, load_video_mode='full', with_ab=True),
        batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

    model = Derendering(n_keypoints=args.n_keypoints,
                        mode=args.mode,
                        n_coefficients=args.n_coefficients).to(device)

    state_dict = torch.load(args.model_weight_path, map_location=torch.device)
    model.load_state_dict(state_dict)
    model.eval()

    dataloc = args.output_path
    os.makedirs(dataloc, exist_ok=True)

    with torch.no_grad():
        for loader in [test_dataloader, val_dataloader, train_dataloader]:
            for i, x in enumerate(tqdm(loader)):
                B, T, C, H, W = x['rgb_ab'].shape
                rgb_ab = x['rgb_ab'].to(device).view(-1, 3, 112, 112)
                rgb_cd = x['rgb_cd'].to(device).view(-1, 3, 112, 112)

                features_ab, heatmap_ab, parameter_ab = model.encoder(rgb_ab)
                parameter_ab = model.extractor(torch.flatten(parameter_ab, start_dim=2))
                parameter_ab = torch.sigmoid(parameter_ab)
                keypoints_ab = model.get_keypoint_location(heatmap_ab)

                state_ab = torch.cat([keypoints_ab, parameter_ab], dim=-1)

                features_cd, heatmap_cd, parameter_cd = model.encoder(rgb_cd)
                parameter_cd = model.extractor(torch.flatten(parameter_cd, start_dim=2))
                parameter_cd = torch.sigmoid(parameter_cd)
                keypoints_cd = model.get_keypoint_location(heatmap_cd)
                state_cd = torch.cat([keypoints_cd, parameter_cd], dim=-1)

                state_ab = state_ab.view(-1, T, state_ab.shape[-2], state_ab.shape[-1])
                state_cd = state_cd.view(-1, T, state_ab.shape[-2], state_ab.shape[-1])

                state_ab = state_ab.detach().cpu().numpy()
                state_cd = state_cd.detach().cpu().numpy()

                for i in range(x["ex"].shape[0]):
                    ex = x["ex"][i].detach().cpu().numpy()
                    try:
                        os.makedirs(dataloc + "/" + str(ex), exist_ok=True)
                    except FileExistsError:
                        pass
                    np.save(dataloc + "/" + str(ex) + "/keypoints_ab.npy", state_ab[i])
                    np.save(dataloc + "/" + str(ex) + "/keypoints_cd.npy", state_cd[i])


if __name__ == '__main__':
    main()
