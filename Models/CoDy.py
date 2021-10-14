import torch
import torch.nn as nn
from Models.modules import GCN, MLP


class ConfoundersModule(nn.Module):
    def __init__(self, state_size, cf_size, n_layers, hidden_size, emb_size):
        super(ConfoundersModule, self).__init__()

        self.gn_ab = GCN(in_size=state_size, out_size=emb_size, n_layers=n_layers, hidden_size=hidden_size,
                         batchnorm=False)
        self.rnn_ab = nn.GRU(emb_size, cf_size, num_layers=2, batch_first=True)

    def forward(self, x):
        B, T, K, S = x.shape

        embeddings = self.gn_ab(x.view(B * T, K, S)).view(B, T, K, -1)
        embeddings = embeddings.transpose(1, 2).reshape(B * K, T, -1)
        output, _ = self.rnn_ab(embeddings)
        U = output.view(B, K, T, -1)[:, :, -1]
        return U


class DynamicModule(nn.Module):
    def __init__(self, state_size, cf_size, n_layers, hidden_size, emb_size, n_gn):
        super(DynamicModule, self).__init__()

        gns = [GCN(in_size=state_size + cf_size, out_size=emb_size, n_layers=n_layers, hidden_size=hidden_size,
                   batchnorm=False)]
        for _ in range(1, n_gn):
            gns.append(GCN(in_size=emb_size, out_size=emb_size, n_layers=n_layers, hidden_size=hidden_size,
                           batchnorm=False))

        self.gn_cd = nn.ModuleList(gns)
        self.rnn_cd = nn.GRU(emb_size, emb_size, num_layers=2, batch_first=True)
        self.linear_cd = nn.Linear(emb_size, state_size, bias=False)

    def forward(self, inpt, hidden_cd):
        B, K, S = inpt.shape

        embedding = self.gn_cd[0](inpt)
        for i in range(1, len(self.gn_cd) - 1):
            embedding = self.gn_cd[i](embedding)

        if len(self.gn_cd) > 1:
            embedding = self.gn_cd[-1](embedding)

        embedding = embedding.view(B * K, 1, -1)

        if hidden_cd is None:
            output, hidden_cd = self.rnn_cd(embedding)
        else:
            output, hidden_cd = self.rnn_cd(embedding, hidden_cd)
        delta = self.linear_cd(output[:, 0]).view(B, K, -1)
        return delta, hidden_cd


class Encoder(nn.Module):
    def __init__(self, state_size, z_size):
        super(Encoder, self).__init__()
        self.gn_encoder = GCN(in_size=state_size, out_size=z_size, n_layers=0, hidden_size=32)

    def forward(self, x):
        B, T, K, S = x.shape
        z = self.gn_encoder(x.view(B * T, K, S)).view(B, T, K, -1)
        return z


class Decoder(nn.Module):
    def __init__(self, state_size, z_size):
        super(Decoder, self).__init__()
        self.gn_decoder = GCN(in_size=z_size, out_size=state_size, n_layers=0, hidden_size=32)
        self.rnn_decoder = nn.GRU(input_size=z_size, hidden_size=z_size, num_layers=1, batch_first=True)

    def forward(self, x):
        B, T, K, S = x.shape
        x = x.transpose(1, 2).reshape(B * K, T, -1)
        z, _ = self.rnn_decoder(x)
        z = z.view(B, K, T, -1).transpose(1, 2)
        x = self.gn_decoder(z.reshape(B * T, K, -1)).reshape(B, T, K, -1)
        return x


class CoDy(nn.Module):
    def __init__(self, state_size=14, cf_size=32, n_layers=2, hidden_size=32, emb_size=None, n_gn=1, z_size=64):
        super().__init__()

        if emb_size is None:
            emb_size = hidden_size

        self.confounders_module = ConfoundersModule(state_size, cf_size, n_layers=n_layers, hidden_size=hidden_size,
                                                    emb_size=emb_size)
        self.dynamic_module = DynamicModule(z_size, cf_size, n_layers=n_layers, hidden_size=hidden_size,
                                            emb_size=emb_size, n_gn=n_gn)
        self.encoder = Encoder(state_size, z_size)
        self.decoder = Decoder(state_size, z_size)

    def forward(self, ab, c, horizon=149, cf=None):
        if cf is None:
            cf = self.confounders_module(ab)

        list_z = [self.encoder(c.unsqueeze(1)).squeeze(1)]
        hidden_cd = None

        for t in range(horizon):
            inpt = torch.cat([list_z[-1], cf], dim=-1)
            delta, hidden_cd = self.dynamic_module(inpt, hidden_cd)
            pose = list_z[-1] + delta

            list_z.append(pose)

        z = torch.stack(list_z, dim=1)
        cd_hat = self.decoder(z)
        return cd_hat