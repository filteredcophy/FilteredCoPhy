from torch.utils.data import Dataset
import os
import torch


class collisionCF_Keypoints(Dataset):
    def __init__(self, mode="train", model="derendering", n_kpts=4):
        super(collisionCF_Keypoints, self).__init__()

        self.dataloc = f"collision/{n_kpts}/{model}/"
        assert os.path.isdir(self.dataloc)

        with open(f"Datasets/collisionCF_{mode}", "r") as file:
            self.index = [int(k) for k in file.readlines()]

        self.states_ab = torch.load(self.dataloc + f"{mode}/states_ab.tsr")
        self.states_cd = torch.load(self.dataloc + f"{mode}/states_cd.tsr")

        self.state_size = self.states_cd.shape[-1]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        ex = self.index[item]
        keypoints_ab = self.states_ab[item]
        keypoints_cd = self.states_cd[item]
        out = {"keypoints_ab": keypoints_ab, "keypoints_cd": keypoints_cd, 'ex': ex}
        return out