import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_size, h_size, out_size, n_layers, activation=nn.ReLU, batchnorm=False, n_channel=4):
        super(MLP, self).__init__()

        layer_list = []
        if batchnorm:
            layer_list.append(nn.BatchNorm1d(n_channel))

        layer_list.append(nn.Linear(in_size, h_size))
        layer_list.append(activation())
        for _ in range(n_layers):
            layer_list.append(nn.Linear(h_size, h_size))
            layer_list.append(activation())

        layer_list.append(nn.Linear(h_size, out_size))

        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)


class GCN(nn.Module):
    def __init__(self, in_size, out_size, n_layers, hidden_size, activation=nn.ReLU, batchnorm=False):
        super(GCN, self).__init__()

        self.f = MLP(2 * in_size, hidden_size, hidden_size, n_layers, activation, batchnorm, n_channel=2 * in_size)
        self.g = MLP(2 * hidden_size + in_size, hidden_size, out_size, n_layers, activation, batchnorm, n_channel=4)

    def forward(self, x):
        B, K, S = x.shape

        x1 = x.unsqueeze(1).repeat(1, K, 1, 1)
        x2 = x.unsqueeze(2).repeat(1, 1, K, 1)

        x12 = torch.cat([x1, x2], dim=-1)
        interactions = self.f(x12.view(B * K * K, 2 * S)).view(B, K, K, -1)
        E = interactions * (1 - torch.eye(K).view(1, K, K, 1).repeat(B, 1, 1, interactions.shape[-1]).to(x.device))

        E = E.sum(2)
        E_sum = E.sum(1).unsqueeze(1).repeat(1, K, 1)

        out = self.g(torch.cat([x, E, E_sum], dim=-1))

        return out
