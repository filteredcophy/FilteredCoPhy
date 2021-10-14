from torch.utils.data import Dataset
import numpy as np
import torchvision
import os
import cv2
from random import randint
import torch
from tqdm import tqdm
import pybullet as pb


class blocktowerCF_Video(Dataset):
    def __init__(self, mode="train", n_objects=4, resolution=112, with_cd=True, load_video_mode="rand", mono=False,
                 with_ab=False, with_state=False):
        super(blocktowerCF_Video, self).__init__()
        assert load_video_mode in ['rand', 'fix', "full"]

        self.dataloc = f"blocktowerCF/{n_objects}/"
        assert os.path.isdir(self.dataloc)

        with open(f"Datasets/blocktowerCF_{n_objects}_{mode}", "r") as file:
            self.index = [int(k) for k in file.readlines()]

        self.resolution = resolution
        self.with_cd = with_cd
        self.load_video_mode = load_video_mode
        self.mono = mono
        self.with_ab = with_ab
        self.with_state = with_state

        self.rgb_ab = []
        self.rgb_cd = []

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        ex = self.index[item]
        out = {'ex': ex}
        if self.with_ab:
            ab = self.dataloc + str(ex) + "/ab/"
            rgb_ab, r_ab = get_rgb(ab + "rgb.mp4", self.load_video_mode, self.mono)
            out['rgb_ab'] = rgb_ab

        if self.with_cd:
            cd = self.dataloc + str(ex) + "/cd/"
            rgb_cd, r_cd = get_rgb(cd + 'rgb.mp4', self.load_video_mode, self.mono)
            out['rgb_cd'] = rgb_cd

        if self.with_state:
            states = np.load(self.dataloc + str(ex) + '/cd/states.npy')
            viewMatrix = np.array(pb.computeViewMatrix([0, -7, 4.5], [0, 0, 1.5], [0, 0, 1])).reshape(
                (4, 4)).transpose()
            projectionMatrix = np.array(pb.computeProjectionMatrixFOV(60, 112 / 112, 4, 20)).reshape((4, 4)).transpose()
            positions = states[..., :3]
            pose_2d = []
            for t in range(positions.shape[0]):
                pose_2d.append([])
                for k in range(positions.shape[1]):
                    if not np.all(positions[t, k] == 0):
                        pose_2d[-1].append(convert_to_2d(positions[t, k], viewMatrix, projectionMatrix, 112))
                    else:
                        pose_2d[-1].append(np.zeros(2))
            pose_2d = np.array(pose_2d)
            out["pose_2D_cd"] = pose_2d[r_cd, :, :]
        return out


def convert_to_2d(pose, view, projection, resolution):
    center_pose = np.concatenate([pose[:3], np.ones((1))], axis=-1).reshape((4, 1))
    center_pose = view @ center_pose
    center_pose = projection @ center_pose
    center_pose = center_pose[:3] / center_pose[-1]
    center_pose = (center_pose + 1) / 2 * resolution
    center_pose[1] = resolution - center_pose[1]
    return center_pose[:2].astype(int).flatten()


def get_rgb(filedir, load_video_mode, mono):
    if load_video_mode == "full":
        rgb, _, _ = torchvision.io.read_video(filedir, pts_unit="sec")
        rgb = 2 * (rgb / 255) - 1
        rgb = rgb.permute(0, 3, 1, 2)
        r = list(range(150))
    else:
        t = randint(0, 124) if load_video_mode == "rand" else 25
        r = [t, t + 25] if mono else range(t, t + 25)
        capture = cv2.VideoCapture(filedir)
        list_rgb = []
        for i in r:
            capture.set(1, i)
            ret, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            list_rgb.append(frame)
        rgb = np.stack(list_rgb, 0)
        rgb = 2 * (rgb / 255) - 1
        rgb = rgb.astype(np.float32).transpose(0, 3, 1, 2)
        rgb = torch.FloatTensor(rgb)
    return rgb, r