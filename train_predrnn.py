import os
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from Dataloader.Video_Loaders import blocktowerCF_Video, ballsCF_Video, collisionCF_Video
import argparse
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import math
from Models.PredRNN import RNN

parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--patchsize', default=4, type=int)
parser.add_argument('--dataset', default="balls", type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--batchsize', default=16, type=int)

parser.add_argument('--reverse_scheduled_sampling', default=1, type=int)
parser.add_argument('--r_sampling_step_1', default=500, type=int)
parser.add_argument('--r_sampling_step_2', default=1000, type=int)
parser.add_argument('--r_exp_alpha', default=2000, type=int)
parser.add_argument('--decouple_beta', default=0.01, type=int)

parser.add_argument('--total_length', default=150, type=int)
parser.add_argument('--input_length', default=1, type=int)
parser.add_argument('--reverse_input', default=1, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--name', default='predRNN', type=str)
args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

MSE = nn.MSELoss()
datasets = {"blocktower": blocktowerCF_Video, "balls": ballsCF_Video, "collision": collisionCF_Video}
SAVE_PATH = f"trained_models/PredRNN/{args.name}"


def reserve_schedule_sampling_exp(itr, B):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (B, args.input_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (B, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((112 // args.patchsize,
                    112 // args.patchsize,
                    args.patchsize ** 2 * 3))  # 3 should be in channels
    zeros = np.zeros((112 // args.patchsize,
                      112 // args.patchsize,
                      args.patchsize ** 2 * 3))  # 3 should be in channels

    real_input_flag = []
    for i in range(B):
        for j in range(args.total_length - 2):
            if j < args.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (B, args.total_length - 2,
                                  112 // args.patchsize,
                                  112 // args.patchsize,
                                  args.patchsize ** 2 * 3))  # 3 should be in channels
    return torch.FloatTensor(real_input_flag).permute([0, 1, 4, 2, 3]).to(device)


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    patch_tensor = patch_tensor.permute([0, 1, 3, 4, 2])
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = channels // (patch_size * patch_size)
    a = patch_tensor.reshape(batch_size, seq_length,
                             patch_height, patch_width,
                             patch_size, patch_size,
                             img_channels)
    b = a.permute(0, 1, 2, 4, 3, 5, 6)
    img_tensor = b.reshape(batch_size, seq_length,
                           patch_height * patch_size,
                           patch_width * patch_size,
                           img_channels)
    return img_tensor.permute([0, 1, 4, 2, 3])


def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    B, T, C, H, W = img_tensor.shape
    a = img_tensor.reshape(B, T, C,
                           H // patch_size, patch_size,
                           W // patch_size, patch_size)
    b = a.permute([0, 1, 4, 6, 2, 3, 5, ])
    patch_tensor = b.reshape(B, T,
                             patch_size * patch_size * C,
                             H // patch_size,
                             W // patch_size, )
    return patch_tensor


def evaluate():
    print(args)
    dataloader = DataLoader(
        datasets[args.dataset](mode='test', resolution=112, sampling_mode="full"),
        batch_size=1, num_workers=1, pin_memory=True, shuffle=False)

    state_dict = torch.load(SAVE_PATH, map_location=device)

    num_hidden = [128, 128, 128, 128]
    model = RNN(len(num_hidden), num_hidden, args).to(device)
    model.load_state_dict(state_dict)
    validate(model, dataloader, viz=True)


def visualize(gt, pred, z):
    import matplotlib.pyplot as plt

    N = 8
    assert N <= gt.shape[1]
    fig, ax = plt.subplots(2, N, figsize=(2 * N, 4))

    indices = np.linspace(0, gt.shape[1] - 1, N).astype(int)

    gt = ((gt[0] + 1) / 2).numpy().transpose([0, 2, 3, 1])
    pred = ((pred[0] + 1) / 2).numpy().transpose([0, 2, 3, 1])

    gt = np.clip(gt, 0, 1)
    pred = np.clip(pred, 0, 1)

    for i in range(N):
        ax[0][i].axis("off")
        ax[0][i].imshow(gt[indices[i]])
        ax[1][i].axis("off")
        ax[1][i].imshow(pred[indices[i]])
        ax[0][i].set_title(str(indices[i]))
    fig.tight_layout()
    fig.savefig(f"moving_kpts/video/predRNN/{args.dataset}/{z}.png")
    plt.clf()
    plt.cla()
    # plt.show()


def validate(model, dataloader, viz=False):
    model.eval()
    list_loss = []
    with torch.no_grad():
        if args.reverse_scheduled_sampling == 1:
            mask_input = 1
        else:
            mask_input = args.input_length

        real_input_flag = torch.zeros(args.batchsize,
                                      args.total_length - mask_input - 1,
                                      args.patchsize ** 2 * 3,
                                      112 // args.patchsize,
                                      112 // args.patchsize, ).to(device)

        if args.reverse_scheduled_sampling == 1:
            real_input_flag[:, :args.input_length - 1, :, :] = 1.0

        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            rgb = x['rgb_cd'].to(device)[:, :args.total_length]
            B, T, C, H, W = rgb.shape
            rgb_input = reshape_patch(rgb, args.patchsize)
            next_frames, loss = model(rgb_input, real_input_flag[:B])
            img_gen = reshape_patch_back(next_frames, args.patchsize)

            loss_rgb = MSE(img_gen, rgb)

            costs = [loss, loss_rgb]

            cd = rgb.reshape(-1, C, H, W).cpu().numpy().transpose(0, 2, 3, 1)
            predictions = img_gen.reshape(-1, C, H, W).cpu().numpy().transpose(0, 2, 3, 1)
            costs.append(
                np.mean(
                    [peak_signal_noise_ratio(cd[b], predictions[b] + 1e-4, data_range=2) for b in range(cd.shape[0])]))
            list_loss.append(costs[0].cpu().detach().numpy())
            if viz:
                visualize(rgb, img_gen, i)
    print("Val Loss : ", np.mean(list_loss))


def main():
    print(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(f"../trained_models/predRNN/{args.dataset}/",
                exist_ok=True)

    train_dataloader = DataLoader(
        datasets[args.dataset](mode='train', resolution=112, sampling_mode="rand"),
        batch_size=args.batchsize, num_workers=1, pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(
        datasets[args.dataset](mode='val', resolution=112, sampling_mode='fix'),
        batch_size=args.batchsize, num_workers=1, pin_memory=True)

    num_hidden = [128, 128, 128, 128]
    model = RNN(len(num_hidden), num_hidden, args).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        model.train()
        print("=== EPOCH ", epoch + 1, " ===")
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            rgb = x['rgb_cd'].to(device)[:, :args.total_length]
            rgb = reshape_patch(rgb, args.patchsize)
            if args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(i, rgb.shape[0])
            else:
                eta, real_input_flag = schedule_sampling(1, i)

            optim.zero_grad()
            next_frames, loss = model(rgb, real_input_flag)
            loss.backward()
            optim.step()

            if args.reverse_input:
                ims_rev = torch.flip(rgb, dims=[1]).clone()
                optim.zero_grad()
                next_frames, loss_reversed = model(ims_rev, real_input_flag)
                loss_reversed.backward()
                optim.step()
            else:
                loss_reversed = torch.zeros(1)

        validate(model, val_dataloader)
        torch.save(model.state_dict(), SAVE_PATH)


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
