import torch
import argparse
from Dataloader.Keypoints_BlocktowerCF import blocktowerCF_Keypoints
from Dataloader.Keypoints_BallsCF import ballsCF_Keypoints
from Dataloader.Keypoints_CollisionCF import collisionCF_Keypoints
import torch.nn as nn
from Models.VCDN_Keypoints import DynaNetGNN
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--k', default=16, type=int)
parser.add_argument('--nf', default=16 * 4, type=int)
parser.add_argument('--mode', default="long", type=str)
parser.add_argument('--dataset', default="blocktower", type=str)
parser.add_argument('--name', default='vcdn', type=str)
args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

MSE = nn.MSELoss()
BATCHSIZE = 32
datasets = {"blocktower": blocktowerCF_Keypoints, "balls": ballsCF_Keypoints, "collision": collisionCF_Keypoints}


def evaluate():
    dataloader = DataLoader(
        datasets[args.dataset](mode='test', model=f"derendering_{args.k}kpts_nocoef", n_kpts=args.k),
        batch_size=16,
        shuffle=True,
        num_workers=1,
        pin_memory=True)
    model = DynaNetGNN(k=args.k, nf=args.nf, use_batchnorm=True).to(device)
    model.load_state_dict(
        torch.load(f"trained_models/VCDN/{args.dataset}/{args.name}.nn", map_location=device))

    validate(model, dataloader, viz=False)


def visualize(state, predicted_state):
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    for k in range(state.shape[2]):
        ax1.plot(state[0, :, k, 1], -state[0, :, k, 0], c="black", marker=".")
        ax1.plot(predicted_state[0, :, k, 1], -predicted_state[0, :, k, 0], marker=".")

    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    plt.tight_layout()
    plt.show()


def validate(model, dataloader, viz=False, epoch=1000):
    model.eval()
    list_loss = []
    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            ab = x["keypoints_ab"].to(device)[..., :2]
            cd = x["keypoints_cd"].to(device)[..., :2]

            if args.mode == "long":
                c = cd[:, :1]
                horizon = ab.shape[1] - 1
                target = cd[:, 1:]
            else:
                t = random.randint(0, ab.shape[1] - 30)
                c = cd[:, t:t + 5]
                target = cd[:, t + 5:t + 15]
                horizon = 10

            graph, cd_hat = model(ab, c, horizon=horizon)
            cost = get_loss(target, cd_hat, graph)
            list_loss.append(cost[0].detach().cpu().numpy())
            if viz:
                visualize(cd, cd_hat)
    print("Val Loss : ", np.mean(list_loss))
    return np.mean(list_loss)


def Hloss(x):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    return b.sum(1).mean()


def get_loss(cd, pred, graph):
    edge_type = graph["edges_type_logits"]
    loss_H = Hloss(edge_type.view(-1, edge_type.shape[-1]))

    B, T, K, S = pred.shape
    covar_gt = torch.FloatTensor([5e-2, 0, 0, 5e-2]).to(cd.device)
    covar_gt = covar_gt.view(1, 1, 1, -1).repeat(B, T, K, 1).view(-1, K, 2, 2)
    cd_reshaped = cd.reshape(B * T, K, -1)
    mean_pred, covar_pred = pred[..., :2].view(-1, K, 2), pred[..., 2:].view(B * T, K, 2, 2)
    m_gt = MultivariateNormal(cd_reshaped, scale_tril=covar_gt)
    m_pred = MultivariateNormal(mean_pred, scale_tril=covar_pred)

    loss_prob = (m_gt.log_prob(cd_reshaped) - m_pred.log_prob(cd_reshaped)).mean()

    loss_mse = MSE(cd, pred[..., :2])

    loss = loss_prob * 10 + loss_H
    return [loss, loss_prob, loss_H, loss_mse]


def main():
    print(args)
    torch.manual_seed(3)
    random.seed(7)
    np.random.seed(21)
    batchsize = 8

    train_dataloader = DataLoader(
        datasets[args.dataset](mode='train', model=f"derendering_{args.k}kpts_nocoef", n_kpts=args.k),
        batch_size=batchsize,
        shuffle=True,
        num_workers=1,
        pin_memory=True)

    val_dataloader = DataLoader(
        datasets[args.dataset](mode='val', model=f"derendering_{args.k}kpts_nocoef", n_kpts=args.k),
        batch_size=batchsize,
        shuffle=True,
        num_workers=1,
        pin_memory=True)

    model = DynaNetGNN(k=args.k, nf=args.nf, use_batchnorm=True).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.6, patience=2, verbose=True)

    for epoch in range(args.epoch):
        model.train()
        print("=== EPOCH ", epoch + 1, " ===")
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            ab = x["keypoints_ab"].to(device)[..., :2]
            cd = x["keypoints_cd"].to(device)[..., :2]

            if args.mode == "long":
                c = cd[:, :1]
                horizon = ab.shape[1] - 1
                target = cd[:, 1:]
            else:
                t = random.randint(0, ab.shape[1] - 30)
                c = cd[:, t:t + 5]
                target = cd[:, t + 5:t + 15]
                horizon = 10

            graph, cd_hat = model(ab, c, horizon=horizon)
            cost = get_loss(target, cd_hat, graph)
            optim.zero_grad()
            cost[0].backward()
            optim.step()

        error = validate(model, val_dataloader, epoch=epoch)
        sched.step(error)
        torch.save(model.state_dict(), f"../trained_models/VCDN/{args.dataset}/{args.name}.nn")
        print("Saved!")

    evaluate()


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
