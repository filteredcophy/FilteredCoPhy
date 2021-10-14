import torch
import argparse
from Dataloader.Keypoints_BlocktowerCF import blocktowerCF_Keypoints
from Dataloader.Keypoints_BallsCF import ballsCF_Keypoints
from Dataloader.Keypoints_CollisionCF import collisionCF_Keypoints
import torch.nn as nn
from Models.CoDy import CoDy
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--dataset', default="collision", type=str)
parser.add_argument('--keypoints_model', default="derendering", type=str)
parser.add_argument('--n_keypoints', default=4, type=int)
parser.add_argument('--embedding_size', default=16, type=int)
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--n_layers', default=2, type=int)
parser.add_argument('--name', default='cophy', type=str)
args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

MSE = nn.MSELoss()
datasets = {"blocktower": blocktowerCF_Keypoints, "balls": ballsCF_Keypoints, "collision": collisionCF_Keypoints}


# logger.disable_save()

def evaluate():
    dataloader = DataLoader(datasets[args.dataset](mode='test', model=args.keypoints_model, n_kpts=args.n_keypoints),
                            batch_size=16, shuffle=False)

    model = CoDy(state_size=dataloader.dataset.state_size,
                 z_size=args.embedding_size,
                 cf_size=32,
                 emb_size=128,
                 n_layers=args.n_layers,
                 n_gn=1,
                 hidden_size=args.hidden_size).to(device)
    model.load_state_dict(
        torch.load(f"trained_models/cophynet/{args.dataset}/{args.name}.nn", map_location=device))
    validate(model, dataloader, viz=False)


def visualize(state, predicted_state):
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    axs = [fig.add_subplot(2, 4, 3), fig.add_subplot(2, 4, 4), fig.add_subplot(2, 4, 7), fig.add_subplot(2, 4, 8)]
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    for k in range(4):
        ax1.plot(state[0, :, k, 1], -state[0, :, k, 0], c="black", marker=".")
        ax1.plot(predicted_state[0, :, k, 1], -predicted_state[0, :, k, 0], marker=".")

        for i in range(4):
            axs[k].plot(state[0, :100, k, 2 + i], label="Alpha " + str(i), c=colors[i])
            axs[k].plot(predicted_state[0, :100, k, 2 + i], '--', c=colors[i])
            axs[k].set_ylim(0, 1)

        axs[k].legend()

    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    plt.tight_layout()
    plt.show()


def validate(model, dataloader, viz=False, epoch=1000):
    model.eval()
    S = dataloader.dataset.state_size
    list_loss = []
    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            ab = x["keypoints_ab"].to(device)[..., :S]
            cd = x["keypoints_cd"].to(device)[..., :S]
            c = cd[:, 0]
            state_hat = model(ab, c, horizon=ab.shape[1] - 1)

            autoencode = model.decoder(model.encoder(cd))
            cost = get_loss(cd, state_hat, autoencode)
            list_loss.append(cost[0].cpu().detach().numpy())
            if viz:
                visualize(cd, state_hat)
    print("Val Loss : ", np.mean(list_loss))
    return np.mean(list_loss)


def get_loss(cd, cd_hat, autoencode):
    loss_kpts = MSE(cd[..., :2], cd_hat[..., :2])
    loss_coef = MSE(cd[..., 2:7], cd_hat[..., 2:7])
    loss_speed = MSE(cd[..., :7], cd_hat[..., :7])

    loss_encoding = MSE(cd, autoencode)
    loss = MSE(cd, cd_hat) * 1e3 + loss_encoding * 1e3
    return [loss, loss_kpts, loss_coef, loss_speed, loss_encoding]


def main():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    batchsize = 64

    os.makedirs(f"../trained_models/cophynet/{args.dataset}", exist_ok=True)

    train_dataloader = DataLoader(
        datasets[args.dataset](mode='train', model=args.keypoints_model, n_kpts=args.n_keypoints),
        batch_size=batchsize,
        shuffle=True,
        num_workers=1,
        pin_memory=True)
    val_dataloader = DataLoader(datasets[args.dataset](mode='val', model=args.keypoints_model, n_kpts=args.n_keypoints),
                                batch_size=batchsize,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True)
    S = train_dataloader.dataset.state_size
    model = CoDy(state_size=S,
                 z_size=args.embedding_size,
                 cf_size=32,
                 emb_size=128,
                 n_layers=2,
                 n_gn=1,
                 hidden_size=64).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=10, verbose=True,
                                                       min_lr=1e-5)

    for epoch in range(args.epoch):
        model.train()
        print("=== EPOCH ", epoch + 1, " ===")
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            ab = x["keypoints_ab"].to(device)[..., :S]
            cd = x["keypoints_cd"].to(device)[..., :S]

            c = cd[:, 0]
            state_hat = model(ab, c, horizon=ab.shape[1] - 1)

            autoencode = model.decoder(model.encoder(cd))
            cost = get_loss(cd, state_hat, autoencode)

            optim.zero_grad()
            cost[0].backward()
            optim.step()

        error = validate(model, val_dataloader, epoch=epoch)
        sched.step(error)
        torch.save(model.state_dict(), f"../trained_models/cophynet/{args.dataset}/{args.name}.nn")
        print("Saved!")

    evaluate()


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
