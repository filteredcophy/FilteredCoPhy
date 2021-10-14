import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Derendering(nn.Module):
    def __init__(self, n_keypoints=4, n_coefficients=4, mode="fixed", std=0.1, device=torch.device("cpu")):
        super(Derendering, self).__init__()
        assert mode in ["fixed", "learned"]
        self.encoder = Encoder(k=n_keypoints)
        self.decoder = Decoder(k=n_keypoints, n_param=n_coefficients)
        self.extractor = nn.Sequential(
            nn.Linear(28 * 28, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, n_coefficients + 1)
        )

        self.mode = mode
        self.std = std
        self.k = n_keypoints
        self.n_coefficients = n_coefficients

        if n_coefficients > 0:
            if mode == "fixed":
                self.bank = get_dilatation_filters(5, 5, n_coefficients)
                self.bank = self.bank.to(device).view(1, 1, self.n_coefficients, 5, 5)
                self.bank = self.bank.repeat(1, self.k, 1, 1, 1)
            elif mode == "learned":
                self.bank = nn.Parameter(torch.randn(1, 1, n_coefficients, 5, 5).repeat(1, self.k, 1, 1, 1) / 5,
                                         requires_grad=True)

    def forward(self, source, target):
        out = {}
        B, C, H, W = source.shape
        features, heatmap, coef = self.encoder(torch.cat([source, target], dim=0))

        source_features = features[:B]
        target_heatmap = heatmap[B:]
        target_coef = coef[B:]

        target_parameter = self.extractor(torch.flatten(target_coef, start_dim=2))
        target_parameter = torch.sigmoid(target_parameter)

        target_keypoints_location = self.get_keypoint_location(target_heatmap)
        target = self.create_img(source_features, target_keypoints_location, target_parameter)

        out['source_features'] = source_features
        out["target_keypoints"] = target_keypoints_location
        out["target_parameter"] = target_parameter
        out['target'] = target

        return out

    def get_keypoint_location(self, keypoints_features):
        normalized_keypoints = spatial_softmax(keypoints_features)
        S_row = normalized_keypoints.sum(-1)  # N, K, H
        S_col = normalized_keypoints.sum(-2)  # N, K, W

        u_row = S_row.mul(torch.linspace(-1, 1, S_row.size(-1), device=normalized_keypoints.device)).sum(-1)
        u_col = S_col.mul(torch.linspace(-1, 1, S_col.size(-1), device=normalized_keypoints.device)).sum(-1)
        return torch.stack((u_row, u_col), -1)

    def create_img(self, features, keypoints_locations, parameters):
        B, C, W, H = features.shape

        target_keypoints = gaussian_map(keypoints_locations, W, H, std=self.std)

        if self.n_coefficients != 0:
            bank = self.bank.repeat(B, 1, 1, 1, 1)

            weighted_bank = bank * parameters[:, :, :-1].view(B, self.k, self.n_coefficients, 1, 1)
            weighted_bank = parameters[:, :, -1].view(B, self.k, 1, 1, 1) * weighted_bank
            weighted_bank = weighted_bank.view(B * self.k * self.n_coefficients, 1, 5, 5)

            target_keypoints = target_keypoints.unsqueeze(2).repeat(1, 1, self.n_coefficients, 1, 1)
            target_keypoints = target_keypoints.view(1, B * self.k * self.n_coefficients, 28, 28)

            keypoints_map = F.conv2d(target_keypoints, weighted_bank, padding=2,
                                     groups=B * self.k * self.n_coefficients)
            keypoints_map = keypoints_map.view(B, -1, 28, 28)
        else:
            keypoints_map = target_keypoints

        inpt = torch.cat([features, keypoints_map], dim=1)
        target_images = self.decoder(inpt)

        return target_images

    def get_features_only(self, x):
        B, C, H, W = x.shape
        x = self.net(x)
        x_feat = self.net_feat(x)
        return x_feat

    def get_keypoints_only(self, x):
        x = self.net(x)
        x_pose = self.net_pose(x)
        x_coef = self.net_coef(x)
        keypoints = F.softplus(self.regressor(x_pose))  # [B, K, 28, 28]
        target_parameter = self.extractor(torch.flatten(x_coef, start_dim=2))
        target_parameter = torch.sigmoid(target_parameter)
        target_keypoints_location = self.get_keypoint_location(keypoints)
        return target_keypoints_location, target_parameter


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.relu(x)


class Encoder(nn.Module):
    def __init__(self, k):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            Block(3, 32, kernel_size=(7, 7), stride=1, padding=3),
            Block(32, 32, kernel_size=(3, 3), stride=1),
            Block(32, 64, kernel_size=(3, 3), stride=2),
            Block(64, 64, kernel_size=(3, 3), stride=1),
            Block(64, 128, kernel_size=(3, 3), stride=2)
        )
        self.net_feat = Block(128, 16, kernel_size=(3, 3), stride=1)

        self.net_pose = Block(128, 128, kernel_size=(3, 3), stride=1)

        self.net_coef = Block(128, k, kernel_size=(3, 3), stride=1)

        self.regressor = nn.Conv2d(128, k, kernel_size=(1, 1))

    def forward(self, x):
        x = self.net(x)  # [B, 128, 28, 28]

        x_feat = self.net_feat(x)  # [B, 128, 28, 28]
        x_pose = self.net_pose(x)  # [B, 128, 28, 28]
        x_coef = self.net_coef(x)

        pose = F.softplus(self.regressor(x_pose))  # [B, K, 28, 28]
        return x_feat, pose, x_coef


class Decoder(nn.Module):
    def __init__(self, k, n_param):
        super(Decoder, self).__init__()
        if n_param == 0:
            i = k
        else:
            i = k * n_param
        self.net = nn.Sequential(
            Block(16 + i, 128, kernel_size=(3, 3), stride=1),  # 6
            Block(128, 128, kernel_size=(3, 3), stride=1),
            Block(128, 64, kernel_size=(3, 3), stride=1),  # 5
            nn.UpsamplingBilinear2d(scale_factor=2),
            Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
            Block(64, 32, kernel_size=(3, 3), stride=1),  # 3
            nn.UpsamplingBilinear2d(scale_factor=2, ),
            Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
            Block(32, 32, kernel_size=(7, 7), stride=1, padding=3),  # 1
            nn.Conv2d(32, 3, 1)
        )

    def forward(self, x):
        return torch.tanh(self.net(x))


def spatial_softmax(features):
    features_reshape = features.reshape(features.shape[:-2] + (-1,))
    output = F.softmax(features_reshape, dim=-1)
    output = output.reshape(features.shape)
    return output


def gaussian_map(mu, width, height, std):
    # features: (N, K, 2)
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    y = torch.linspace(-1.0, 1.0, height, device=mu.device)
    x = torch.linspace(-1.0, 1.0, width, device=mu.device)
    mu_y, mu_x = mu_y.unsqueeze(-1), mu_x.unsqueeze(-1)

    y = y.reshape(1, 1, height, 1)
    x = x.reshape(1, 1, 1, width)

    dist = ((y - mu_y) ** 2 + (x - mu_x) ** 2) / std ** 2
    g_yx = torch.exp(-dist)

    return g_yx


def get_dilatation_filters(W, H, n_params):
    X = torch.linspace(-5, 5, W)
    Y = torch.linspace(-5, 5, H)
    Z = torch.zeros(n_params, W, H)
    for k, theta in enumerate(torch.arange(0, n_params) * (np.pi) / n_params):
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                x0 = (x + y * torch.tan(theta)) / (1 + torch.tan(theta) ** 2)
                y0 = x0 * torch.tan(theta)

                Z[k, i, j] = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2))
    return Z
