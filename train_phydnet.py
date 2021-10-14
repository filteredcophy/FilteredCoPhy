import torch
import argparse
from Dataloader.Video_BlocktowerCF import blocktowerCF_Video
from Dataloader.Video_BallsCF import ballsCF_Video
from Dataloader.Video_CollisionCF import collisionCF_Video
import torch.nn as nn
from Models.PhyDNet import ConvLSTM, PhyCell, EncoderRNN
from Models.constrain_moments import K2M
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--dataset', default="collision", type=str)
parser.add_argument('--mode', default="long", type=str)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--layer_size', default=49, type=int)
parser.add_argument('--name', default='phydnet', type=str)
args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

MSE = nn.MSELoss()
BATCHSIZE = 12

datasets = {"blocktower": blocktowerCF_Video, "balls": ballsCF_Video, "collision": collisionCF_Video}
constraints = torch.zeros((49, 7, 7)).to(device)
ind = 0
for i in range(0, 7):
    for j in range(0, 7):
        constraints[ind, i, j] = 1
        ind += 1


# logger.disable_save()

def evaluate():
    dataloader = DataLoader(
        datasets[args.dataset](mode='test', resolution=112, load_video_mode="full",
                               with_cd=True, with_ab=False), batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True)
    phycell = PhyCell(input_shape=(28, 28), input_dim=64, F_hidden_dims=[args.layer_size] * args.n_layers,
                      n_layers=args.n_layers, kernel_size=(7, 7), device=device)
    convcell = ConvLSTM(input_shape=(28, 28), input_dim=64, hidden_dims=[128, 64], n_layers=2, kernel_size=(3, 3),
                        device=device)
    model = EncoderRNN(phycell, convcell, device)
    model.load_state_dict(
        torch.load(f"trained_models/PhyDNet/{args.dataset}/{args.name}.nn", map_location=device))

    validate(model, dataloader, viz=True)


def visualize(cd, cd_hat):
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")

    cd = (cd[0].numpy().transpose(0, 2, 3, 1) + 1) / 2
    cd_hat = (cd_hat[0].numpy().transpose(0, 2, 3, 1) + 1) / 2
    fig, ax = plt.subplots(2, 8, figsize=(16, 4))

    idx = np.linspace(0, cd.shape[0] - 1, 8).astype(int)
    for k in range(8):
        ax[0][k].imshow(cd[idx[k]])
        ax[1][k].imshow(cd_hat[idx[k]])
        ax[0][k].axis("off")
        ax[1][k].axis("off")
        ax[0][k].set_title(str(idx[k]))
    plt.tight_layout()
    plt.show()


def validate(model, dataloader, viz=False, epoch=1000):
    model.eval()
    list_loss = []
    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            cd = x["rgb_cd"].to(device)

            if args.mode == "short":
                input_image = cd[:, :10]
                target_image = cd[:, 10:]
            else:
                input_image = cd[:, :1]
                target_image = cd[:, 1:]

            cost = torch.zeros(1).to(device)
            for ei in range(input_image.shape[1] - 1):
                encoder_output, encoder_hidden, output_image, _, _ = model(input_image[:, ei, :, :, :], (ei == 0))
                cost += MSE(output_image, input_image[:, ei + 1, :, :, :])

            decoder_input = input_image[:, -1, :, :, :]
            predictions = []
            for di in range(target_image.shape[1]):
                decoder_output, decoder_hidden, output_image, _, _ = model(decoder_input, args.mode == "long")
                decoder_input = output_image
                predictions.append(output_image)

            predictions = torch.stack(predictions, dim=1)
            costs = get_loss(target_image, predictions)
            costs += (10 * torch.log10(4 / costs[1]),)

            list_loss.append(costs[0].detach().cpu().numpy())
            if viz:
                visualize(target_image, predictions)
    print("Val Loss : ", np.mean(list_loss))
    return np.mean(list_loss)


def compute_grad(rgb):
    rgb = rgb.reshape(-1, 3, 112, 112)
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]).expand(1, 3, 3, 3).to(device)
    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]).expand(1, 3, 3, 3).to(device)
    sobel = (F.conv2d(rgb, a, padding=1) ** 2 + F.conv2d(rgb, b, padding=1) ** 2).squeeze(1)
    return sobel


def get_loss(cd, cd_hat):
    loss_rgb = MSE(cd, cd_hat)
    loss_grad = MSE(compute_grad(cd), compute_grad(cd_hat))
    loss = 10_000 * loss_rgb + loss_grad * 1

    return [loss, loss_rgb, loss_grad]


def main():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    os.makedirs(f"../trained_models/PhyDNet/{args.dataset}", exist_ok=True)

    train_dataloader = DataLoader(
        datasets[args.dataset](mode='train', resolution=112, load_video_mode="full",
                               with_cd=True, with_ab=False), batch_size=BATCHSIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    val_dataloader = DataLoader(
        datasets[args.dataset](mode='val', resolution=112, load_video_mode="full",
                               with_cd=True, with_ab=False), batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    phycell = PhyCell(input_shape=(28, 28), input_dim=64, F_hidden_dims=[args.layer_size] * args.n_layers,
                      n_layers=args.n_layers, kernel_size=(7, 7), device=device)
    convcell = ConvLSTM(input_shape=(28, 28), input_dim=64, hidden_dims=[128, 64], n_layers=2, kernel_size=(3, 3),
                        device=device)
    model = EncoderRNN(phycell, convcell, device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=10, factor=0.1, verbose=True)

    for epoch in range(args.epoch):
        model.train()
        teacher_forcing_ratio = np.maximum(0, 1 - epoch * 0.003)
        print("=== EPOCH ", epoch + 1, " ===")
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            cd = x["rgb_cd"].to(device)

            if args.mode == "short":
                t = random.randint(10, 50)
                input_image = cd[:, t - 10:t]
                target_image = cd[:, t:t + 25]
            else:
                input_image = cd[:, :1]
                target_image = cd[:, 1:]
            # print(cd.shape, input_image.shape, target_image.shape)
            cost = torch.zeros(1).to(device)
            for ei in range(input_image.shape[1] - 1):
                encoder_output, encoder_hidden, output_image, _, _ = model(input_image[:, ei, :, :, :], (ei == 0))
                cost += MSE(output_image, input_image[:, ei + 1, :, :, :])

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            decoder_input = input_image[:, -1, :, :, :]
            for di in range(target_image.shape[1]):
                decoder_output, decoder_hidden, output_image, _, _ = model(decoder_input, args.mode == "long")
                target = target_image[:, di, :, :, :]
                cost += MSE(output_image, target)
                if use_teacher_forcing:
                    decoder_input = target  # Teacher forcing
                else:
                    decoder_input = output_image

            # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
            k2m = K2M((7, 7)).to(device)
            for b in range(0, model.phycell.cell_list[0].input_dim):
                filters = model.phycell.cell_list[0].F.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
                m = k2m(filters.double())
                m = m.float()
                cost += MSE(m, constraints)  # constrains is a precomputed matrix

            optim.zero_grad()
            cost.backward()
            optim.step()

        error = validate(model, val_dataloader, epoch=epoch)
        sched.step(error)
        torch.save(model.state_dict(), f"../trained_models/PhyDNet/{args.dataset}/{args.name}.nn")
        print("Saved!")

    evaluate()


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
