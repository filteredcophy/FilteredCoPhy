import torch
import argparse
from Models.VCDN_Derendering import KeyPointNet
from Dataloader.Video_BlocktowerCF import blocktowerCF_Video
from Dataloader.Video_BallsCF import ballsCF_Video
from Dataloader.Video_CollisionCF import collisionCF_Video
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import random
import os
from skimage.metrics import peak_signal_noise_ratio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--dataset', default="collision", type=str)
parser.add_argument('--n_keypoints', default=4, type=int)
parser.add_argument('--name', default='transporter', type=str)
args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

MSE = nn.MSELoss()
BATCHSIZE = 64
datasets = {"blocktower": blocktowerCF_Video, "balls": ballsCF_Video, "collision": collisionCF_Video}


def evaluate():
    print(args)
    dataloader = DataLoader(
        datasets[args.dataset]('test', resolution=112, load_video_mode="fix", mono=True, with_state=True),
        batch_size=1,
        shuffle=False)

    state_dict = torch.load(f"trained_models/VCDN/{args.dataset}/{args.name}.nn",
                            map_location=torch.device('cpu'))

    model = KeyPointNet(n_keypoints=args.n_keypoints, device=device).to(device)
    model.load_state_dict(state_dict)

    validate(model, dataloader, viz=False, epoch=-1)


def vizu(source, target, target_hat, target_keypoints):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    source = source.detach().cpu().numpy()[0].transpose((1, 2, 0))
    target = target.detach().cpu().numpy()[0].transpose((1, 2, 0))
    pred = target_hat.detach().cpu().numpy()[0].transpose((1, 2, 0))

    source = np.clip((source + 1) / 2, 0, 1)
    target = np.clip((target + 1) / 2, 0, 1)
    pred = np.clip((pred + 1) / 2, 0, 1)

    ax1.imshow(source)
    ax2.imshow(target)
    ax3.imshow(target)
    ax4.imshow(pred)

    keypoints_target = target_keypoints.detach().cpu().numpy()[0]

    X, Y = (keypoints_target[:, 0] + 1) / 2 * 112, (keypoints_target[:, 1] + 1) / 2 * 112
    ax3.scatter(X, Y, c="black", s=2)

    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    ax1.set_title("Source")
    ax2.set_title("Target")
    ax3.set_title("Keypoints")
    ax4.set_title("Reconstruction")

    plt.tight_layout()
    plt.show()


def local_psnr(rgb, pred, pose_2d):
    psnr = 0
    size = 10
    for k in range(pose_2d.shape[-2]):
        x, y = int(pose_2d[0, 1, k, 0]), int(pose_2d[0, 1, k, 1])
        if x == 0 and y == 0:
            continue
        if 0 < x < 112 and 0 < y < 112:
            xmax, xmin = min(111, x + size), max(0, x - size)
            ymax, ymin = min(111, y + size), max(0, y - size)
            psnr += peak_signal_noise_ratio(rgb[xmin:xmax, ymin:ymax], pred[xmin:xmax, ymin:ymax], data_range=2)
    return psnr / pose_2d.shape[-2]


def validate(model, dataloader, viz=False, epoch=0):
    model.eval()
    list_loss = []
    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            rgb = x['rgb_cd'].to(device)
            source = rgb[:, 0]
            target = rgb[:, 1]

            out = model(source, target)

            costs = get_loss(target, out['target'])
            cd = target.cpu().numpy().transpose(0, 2, 3, 1)
            predictions = out["target"].cpu().numpy().transpose(0, 2, 3, 1)
            psnr = [peak_signal_noise_ratio(cd[i], predictions[i], data_range=2) for i in range(cd.shape[0])]
            costs.append(np.mean(psnr))

            if epoch == -1:
                costs.append(local_psnr(cd[0], predictions[0], x["pose_2D_cd"]))
            else:
                costs.append(0)

            list_loss.append(costs[0].detach().cpu().numpy())
            if viz:
                vizu(source, target, out["target"], out['target_keypoints'])
        print("Val Loss : ", np.mean(list_loss))
        return np.mean(list_loss)


def compute_grad(rgb):
    rgb = rgb.view(-1, 3, 112, 112)
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]).expand(1, 3, 3, 3).to(device)
    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]).expand(1, 3, 3, 3).to(device)
    sobel = (F.conv2d(rgb, a, padding=1) ** 2 + F.conv2d(rgb, b, padding=1) ** 2).squeeze(1)
    return sobel


def get_loss(target, target_hat):
    loss_rgb = MSE(target_hat, target)
    loss_grad = MSE(compute_grad(target), compute_grad(target_hat))
    loss = 10 * loss_rgb

    return [loss, loss_rgb, loss_grad]


def main():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    os.makedirs(f"../trained_models/VCDN/{args.dataset}", exist_ok=True)

    train_dataloader = DataLoader(datasets[args.dataset]('train', resolution=112, load_video_mode="rand", mono=True),
                                  batch_size=BATCHSIZE,
                                  num_workers=1,
                                  pin_memory=True,
                                  shuffle=True)
    val_dataloader = DataLoader(datasets[args.dataset]('val', resolution=112, load_video_mode='fix', mono=True),
                                batch_size=BATCHSIZE,
                                num_workers=1,
                                pin_memory=True)

    model = KeyPointNet(n_keypoints=args.n_keypoints, device=device).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.6, patience=2, verbose=True)

    for epoch in range(args.epoch):
        model.train()
        print("=== EPOCH ", epoch + 1, " ===")
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            rgb = x['rgb_cd'].to(device)
            source = rgb[:, 0]
            target = rgb[:, 1]

            out = model(source, target)

            cost = get_loss(target, out['target'])

            optim.zero_grad()
            cost[0].backward()
            optim.step()
        error = validate(model, val_dataloader, epoch=epoch)
        sched.step(error)
        torch.save(model.state_dict(), f"../trained_models/VCDN/{args.dataset}/{args.name}.nn")
        print("Saved!")

    validate(model, val_dataloader)


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
