import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, lim=(-1., 1., -1., 1.), temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = nn.Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(lim[0], lim[1], self.width),
            np.linspace(lim[2], lim[3], self.height))

        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self.pos_x) * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self.pos_y) * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel, 2)

        return feature_keypoints


class KeyPointPredictor(nn.Module):
    def __init__(self, k, lim=(-1., 1., -1., 1.)):
        super(KeyPointPredictor, self).__init__()

        nf = 16

        sequence = [
            # input is (ni) x 64 x 64
            nn.Conv2d(3, nf, 7, 1, 3),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # fesrcat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
            nn.Conv2d(nf * 4, k, 1, 1)
            # feat size (n_kp) x 16 x 16
        ]

        self.model = nn.Sequential(*sequence)
        self.integrater = SpatialSoftmax(height=112 // 4, width=112 // 4, channel=k, lim=lim)

    def integrate(self, heatmap):
        return self.integrater(heatmap)

    def forward(self, img):
        heatmap = self.model(img)
        return self.integrate(heatmap)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        nf = 16

        sequence = [
            # input is (ni) x 64 x 64
            nn.Conv2d(3, nf, 7, 1, 3),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, img):
        return self.model(img)


class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        nf = 16

        sequence = [
            # input is (nf * 4) x 16 x 16
            nn.ConvTranspose2d(nf * 4, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.ConvTranspose2d(nf * 2, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf * 2, nf, 5, 1, 2),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf, 3, 7, 1, 3)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, feat):
        return self.model(feat)


class KeyPointNet(nn.Module):
    def __init__(self, n_keypoints, device):
        super(KeyPointNet, self).__init__()

        # visual feature extractor
        self.feature_extractor = FeatureExtractor()

        # key point predictor
        self.keypoint_predictor = KeyPointPredictor(n_keypoints)

        # map the feature back to the image
        self.refiner = Refiner()

        lim = (-1., 1., -1., 1.)
        x = np.linspace(lim[0], lim[1], 112 // 4)
        y = np.linspace(lim[2], lim[3], 112 // 4)
        z = np.linspace(-1., 1., n_keypoints)

        self.x = Variable(torch.FloatTensor(x)).to(device)
        self.y = Variable(torch.FloatTensor(y)).to(device)
        self.z = Variable(torch.FloatTensor(z)).to(device)

    def extract_feature(self, img):
        return self.feature_extractor(img)

    def predict_keypoint(self, img):
        return self.keypoint_predictor(img)

    def keypoint_to_heatmap(self, keypoint, inv_std=10.):
        height = 112 // 4
        width = 112 // 4

        mu_x, mu_y = keypoint[:, :, :1].unsqueeze(-1), keypoint[:, :, 1:].unsqueeze(-1)
        y = self.y.view(1, 1, height, 1)
        x = self.x.view(1, 1, 1, width)

        g_y = (y - mu_y) ** 2
        g_x = (x - mu_x) ** 2
        dist = (g_y + g_x) * inv_std ** 2

        hmap = torch.exp(-dist)

        return hmap

    def transport(self, src_feat, des_feat, src_hmap, des_hmap, des_feat_hmap=None):
        src_hmap = torch.sum(src_hmap, 1, keepdim=True)
        des_hmap = torch.sum(des_hmap, 1, keepdim=True)
        src_digged = src_feat * (1. - src_hmap) * (1. - des_hmap)

        if des_feat_hmap is None:
            mixed_feat = src_digged + des_hmap * des_feat
        else:
            mixed_feat = src_digged + des_feat_hmap

        return mixed_feat

    def refine(self, mixed_feat):
        return self.refiner(mixed_feat)

    def kp_feat(self, feat, hmap):
        B, nf, H, W = feat.size()
        n_kp = hmap.size(1)

        p = feat.view(B, 1, nf, H, W) * hmap.view(B, n_kp, 1, H, W)
        kp_feat = torch.sum(p, (3, 4))
        return kp_feat

    def forward(self, src, des):
        out = {}
        cat = torch.cat([src, des], 0)
        feat = self.extract_feature(cat)
        kp = self.predict_keypoint(cat)
        B = kp.size(0)

        src_feat, des_feat = feat[:B // 2], feat[B // 2:]
        src_kp, des_kp = kp[:B // 2], kp[B // 2:]

        src_hmap = self.keypoint_to_heatmap(src_kp, 10)
        des_hmap = self.keypoint_to_heatmap(des_kp, 10)

        mixed_feat = self.transport(src_feat, des_feat, src_hmap, des_hmap)
        des_pred = self.refine(mixed_feat)

        out['source_features'] = src_feat
        out['target_features'] = des_feat

        out["source_keypoints"] = src_kp
        out["target_keypoints"] = des_kp

        out["source_hmap"] = src_hmap
        out["target_hmap"] = des_hmap

        out['target'] = des_pred

        return out
