from torch.utils.data import Dataset
import os
import numpy as np
import torch


class Keypoints_loader(Dataset):
    def __init__(self, mode="train", path=""):
        super(Keypoints_loader, self).__init__()
        self.index = None
        self.dataloc = path
        assert os.path.isdir(self.dataloc)
        self.state_size = 0
        self.mode = mode

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        ex = self.index[item]
        keypoints_ab = np.load(os.path.join(self.dataloc, str(ex), "keypoints_ab.npy"))
        keypoints_cd = np.load(os.path.join(self.dataloc, str(ex), "keypoints_cd.npy"))

        T, K, S = keypoints_ab.shape
        keypoints_ab = np.concatenate([torch.zeros(1, K, S), keypoints_ab[1:] - keypoints_ab[:-1]], axis=0)
        keypoints_cd = np.concatenate([torch.zeros(1, K, S), keypoints_cd[1:] - keypoints_cd[:-1]], axis=0)

        out = {"keypoints_ab": keypoints_ab, "keypoints_cd": keypoints_cd, 'ex': ex}
        return out


class blocktowerCF_Keypoints(Keypoints_loader):
    def __init__(self, **kwargs):
        super(blocktowerCF_Keypoints, self).__init__(**kwargs)

        with open(f"Datasets/blocktowerCF_4_{self.mode}", "r") as file:
            self.index = [int(k) for k in file.readlines()]
        self.state_size = 14


class ballsCF_Keypoints(Keypoints_loader):
    def __init__(self, **kwargs):
        super(ballsCF_Keypoints, self).__init__(**kwargs)

        with open(f"Datasets/ballsCF_4_{self.mode}", "r") as file:
            self.index = [int(k) for k in file.readlines()]
        self.state_size = 14


class collisionCF_Keypoints(Keypoints_loader):
    def __init__(self, **kwargs):
        super(collisionCF_Keypoints, self).__init__(**kwargs)

        with open(f"Datasets/collisionCF_{self.mode}", "r") as file:
            self.index = [int(k) for k in file.readlines()]
        self.state_size = 14
