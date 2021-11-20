import torch
import argparse
from Dataloader.Keypoints_Loaders import blocktowerCF_Keypoints, ballsCF_Keypoints, collisionCF_Keypoints
import torch.nn as nn
from Models.VCDN_Keypoints import DynaNetGNN
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1, type=int, help="Number of Epoch for training. Can be set to 0 for evaluation")
parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
parser.add_argument('--n_keypoints', default=16, type=int, help="Number of keypoints")
parser.add_argument('--mode', default="long", type=str,
                    help="'long' predicts the entire video from 1 image. "
                         "'short' predicts 10 frames from the 5 previous ones")
parser.add_argument('--dataset', default="blocktower", type=str)
parser.add_argument('--keypoints_path', default="", type=str,
                    help="Datasets, should be one of 'blocktower', 'balls' or 'collision")
parser.add_argument('--name', default='vcdn', type=str, help="Name for weight saving")
args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

MSE = nn.MSELoss()
SAVE_PATH = f"trained_models/VCDN/{args.name}.nn"
BATCHSIZE = 32
datasets = {"blocktower": blocktowerCF_Keypoints, "balls": ballsCF_Keypoints, "collision": collisionCF_Keypoints}


def evaluate():
    dataloader = DataLoader(datasets[args.dataset](mode='test', path=args.keypoints_path),
                            batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

    model = DynaNetGNN(k=args.n_keypoints, nf=4 * 16, use_batchnorm=True).to(device)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
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


def validate(model, dataloader, viz=False):
    model.eval()
    list_loss = []
    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            ab = x["keypoints_ab"].to(device)
            cd = x["keypoints_cd"].to(device)

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

    train_dataloader = DataLoader(datasets[args.dataset](mode='train', path=args.keypoints_path),
                                  batch_size=batchsize, shuffle=True, num_workers=1, pin_memory=True)

    val_dataloader = DataLoader(datasets[args.dataset](mode='val', path=args.keypoints_path),
                                batch_size=batchsize, shuffle=True, num_workers=1, pin_memory=True)

    model = DynaNetGNN(k=args.n_keypoints, nf=4 * 16, use_batchnorm=True).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.6, patience=2, verbose=True)

    for epoch in range(args.epoch):
        model.train()
        print("=== EPOCH ", epoch + 1, " ===")
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            ab = x["keypoints_ab"].to(device)
            cd = x["keypoints_cd"].to(device)

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

        error = validate(model, val_dataloader)
        sched.step(error)
        torch.save(model.state_dict(), SAVE_PATH)
        print("Saved!")

    evaluate()


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
