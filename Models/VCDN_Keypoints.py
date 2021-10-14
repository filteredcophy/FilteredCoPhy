import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_size, out_size, batchnorm=False, n_features=0, n_layers=0):
        super(MLP, self).__init__()
        modules = [nn.Linear(in_size, out_size), nn.ReLU()]
        for _ in range(n_layers):
            modules.append(nn.Linear(out_size, out_size))
            modules.append(nn.ReLU())
        if batchnorm == 1:
            modules.append(nn.BatchNorm1d(n_features))
        self.f = nn.Sequential(*modules)

    def forward(self, x):
        return self.f(x)


class ConfoundersModule(nn.Module):
    def __init__(self, nf, kpts_size, use_batchnorm, k, node_attr_dim, edge_attr_dim, edge_type_num):
        super(ConfoundersModule, self).__init__()
        self.graph_encoder = PropNet(node_dim_in=2,
                                     edge_dim_in=0,
                                     hidden_size=nf * 3,
                                     node_dim_out=nf,
                                     edge_dim_out=nf,
                                     edge_type_num=1,
                                     n_kpts=k,
                                     batch_norm=use_batchnorm)

        self.model_infer_node_agg = CNNet(kernel_size=3, in_channel=nf + kpts_size, hidden_channel=nf * 4,
                                          out_channel=nf)
        self.model_infer_edge_agg = CNNet(kernel_size=3, in_channel=nf + kpts_size * 2, hidden_channel=nf * 4,
                                          out_channel=nf)

        self.model_infer_affi_matx = PropNet(node_dim_in=nf,
                                             edge_dim_in=nf,
                                             hidden_size=nf * 3,
                                             node_dim_out=0,
                                             edge_dim_out=2,
                                             edge_type_num=1,
                                             pstep=2,
                                             n_kpts=k,
                                             batch_norm=use_batchnorm)

        self.model_infer_graph_attr = PropNet(node_dim_in=nf,
                                              edge_dim_in=nf,
                                              hidden_size=nf * 3,
                                              node_dim_out=node_attr_dim,
                                              edge_dim_out=edge_attr_dim,
                                              edge_type_num=edge_type_num,
                                              n_kpts=k,
                                              batch_norm=use_batchnorm)

    def forward(self, kp, hard=False):
        B, T, K, _ = kp.shape
        node_rep, edge_rep = self.graph_encoder(kp.view(B * T, K, -1))
        node_rep = node_rep.view(B, T, K, -1)
        edge_rep = edge_rep.view(B, T, K * K, -1)

        kp_r = kp.unsqueeze(3).repeat(1, 1, 1, K, 1).view(B, T, K * K, -1)
        kp_s = kp.unsqueeze(2).repeat(1, 1, K, 1, 1).view(B, T, K * K, -1)

        node_rep = torch.cat([node_rep, kp], -1)
        edge_rep = torch.cat([edge_rep, kp_r, kp_s], -1)

        node_rep_agg = self.model_infer_node_agg(node_rep).view(B, K, -1)
        edge_rep_agg = self.model_infer_edge_agg(edge_rep).view(B, K, K, -1)

        edge_type_logits = self.model_infer_affi_matx(node_rep_agg, edge_rep_agg, ignore_node=True)

        edge_type = F.gumbel_softmax(edge_type_logits, hard=hard)
        node_attr, edge_attr = self.model_infer_graph_attr(node_rep_agg, edge_rep_agg, edge_type)

        out = {'nodes_attr': node_attr, "edges_attr": edge_attr,
               "edges_type": edge_type, "edges_type_logits": edge_type_logits}
        return out


class DynamicModule(nn.Module):
    def __init__(self, nf, k, node_attr_dim, edge_attr_dim, edge_type_num, drop_prob, use_batchnorm):
        super(DynamicModule, self, ).__init__()
        self.model_dynam_encode = PropNet(node_dim_in=node_attr_dim + 6,
                                          edge_dim_in=edge_attr_dim + 12,
                                          hidden_size=nf * 3,
                                          node_dim_out=nf,
                                          edge_dim_out=nf,
                                          edge_type_num=edge_type_num,
                                          n_kpts=k,
                                          batch_norm=use_batchnorm)

        self.model_dynam_node_forward = GRUNet(nf + 6 + node_attr_dim,
                                               nf * 2, nf,
                                               drop_prob=drop_prob)
        self.model_dynam_edge_forward = GRUNet(nf + 12 + edge_attr_dim,
                                               nf * 2, nf,
                                               drop_prob=drop_prob)

        self.model_dynam_decode = PropNet(node_dim_in=nf + node_attr_dim + 6,
                                          edge_dim_in=nf + edge_attr_dim + 12,
                                          hidden_size=nf * 3,
                                          node_dim_out=5,
                                          edge_dim_out=1,
                                          edge_type_num=edge_type_num,
                                          n_kpts=k,
                                          batch_norm=False)

    def forward(self, kp, graph):
        B, n_his, K, _ = kp.shape
        node_attr = graph["nodes_attr"]
        edge_attr = graph["edges_attr"]
        edge_type = graph["edges_type"]

        node_enc = torch.cat([kp, node_attr.unsqueeze(1).repeat(1, n_his, 1, 1)], dim=-1)
        edge_enc = torch.cat([torch.cat([kp.unsqueeze(3).repeat(1, 1, 1, K, 1),
                                         kp.unsqueeze(2).repeat(1, 1, K, 1, 1)], dim=-1),
                              edge_attr.unsqueeze(1).repeat(1, n_his, 1, 1, 1)], dim=-1)

        node_enc, edge_enc = self.model_dynam_encode(node_enc.view(B * n_his, K, -1),
                                                     edge_enc.view(B * n_his, K, K, -1),
                                                     edge_type[:, None, :, :, :].repeat(1, n_his, 1, 1, 1).view(
                                                         B * n_his, K, K, -1))

        node_enc = node_enc.view(B, n_his, K, -1)
        edge_enc = edge_enc.view(B, n_his, K * K, -1)

        # Dynamic Nodes
        node_enc = torch.cat([kp, node_enc, node_attr.unsqueeze(1).repeat(1, n_his, 1, 1)], dim=-1)
        node_enc = self.model_dynam_node_forward(node_enc).view(B, K, -1)

        # Dynamic Edges
        kp_edge = torch.cat([kp.unsqueeze(3).repeat(1, 1, 1, K, 1), kp.unsqueeze(2).repeat(1, 1, K, 1, 1)], dim=-1)
        kp_edge = kp_edge.view(B, n_his, K ** 2, 12)
        edge_enc = torch.cat([kp_edge, edge_enc, edge_attr.view(B, 1, K * K, -1).repeat(1, n_his, 1, 1)], dim=-1)
        edge_enc = self.model_dynam_edge_forward(edge_enc).view(B, K, K, -1)

        # Decoder
        node_enc = torch.cat([node_enc, node_attr, kp[:, -1]], 2)
        edge_enc = torch.cat([edge_enc, edge_attr, kp_edge[:, -1].view(B, K, K, 12)], dim=-1)
        kp_pred = self.model_dynam_decode(node_enc, edge_enc, edge_type, ignore_edge=True)

        kp_pred = torch.cat([
            kp[:, -1, :, :2] + kp_pred[:, :, :2],  # mean
            F.relu(kp_pred[:, :, 2:3]) + 5e-2,  # covar (0, 0), need to > 0
            torch.zeros(B, K, 1).to(kp.device),  # covar (0, 1)
            kp_pred[:, :, 3:4],  # covar (1, 0)
            F.relu(kp_pred[:, :, 4:5]) + 5e-2],  # covar (1, 1), need to > 0
            dim=2)

        return kp_pred


class DynaNetGNN(nn.Module):
    def __init__(self, k, drop_prob=0.2, use_batchnorm=True, nf=16 * 4):
        super(DynaNetGNN, self).__init__()
        kpts_size = 2

        self.ratio = (112 // 64) * (112 // 64)

        self.node_attr_dim = 1
        self.edge_attr_dim = 1
        self.edge_type_num = 2

        # infer the graph
        self.confounders_module = ConfoundersModule(nf, kpts_size, use_batchnorm, k,
                                                    self.node_attr_dim,
                                                    self.edge_attr_dim,
                                                    self.edge_type_num)

        # dynamics modeling
        self.dynamic_module = DynamicModule(nf, k, self.node_attr_dim,
                                            self.edge_attr_dim, self.edge_attr_dim,
                                            drop_prob=drop_prob, use_batchnorm=use_batchnorm)

    def forward(self, ab, c, horizon=149):
        B, T, K, S = c.shape
        eps = 5e-2
        graph = self.confounders_module(ab)

        covar_gt = torch.FloatTensor([eps, 0, 0, eps]).to(c.device)
        covar_gt = covar_gt.view(1, 1, 1, -1).repeat(B, T, K, 1)
        kp_cur = torch.cat([c, covar_gt], dim=-1)  # B, T, K, 2+4

        list_pred = []
        for j in range(horizon):
            kp_pred = self.dynamic_module(kp_cur, graph)
            list_pred.append(kp_pred)
            kp_cur = torch.cat([kp_cur[:, 1:], kp_pred.unsqueeze(1)], dim=1)

        pred = torch.stack(list_pred, dim=1)
        return graph, pred


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, drop_prob=0.2):
        super(GRUNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, T, K, nf = x.shape
        x = x.transpose(1, 2).reshape(B * K, T, nf)
        out, h = self.gru(x)
        out = self.fc(self.relu(out))
        return out[:, -1]


class CNNet(nn.Module):
    def __init__(self, kernel_size, in_channel, hidden_channel, out_channel, do_prob=0.):
        super(CNNet, self).__init__()

        pool = nn.MaxPool1d(
            kernel_size=2, stride=None, padding=0,
            dilation=1, return_indices=False,
            ceil_mode=False)
        self.f = nn.Sequential(
            nn.Conv1d(in_channel, hidden_channel, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(), nn.BatchNorm1d(hidden_channel),
            nn.Dropout(do_prob), pool,
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(), nn.BatchNorm1d(hidden_channel),
            nn.Dropout(do_prob), pool,
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(), nn.BatchNorm1d(hidden_channel),
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1))

    def forward(self, inputs):
        B, T, K, S = inputs.shape
        x = inputs.transpose(1, 2).reshape(B * K, T, S).transpose(-1, -2)
        pred = self.f(x)
        ret = pred.max(dim=2)[0]
        return ret


class PropNet(nn.Module):
    def __init__(self, node_dim_in, edge_dim_in, hidden_size, node_dim_out, edge_dim_out,
                 edge_type_num=1, pstep=1, batch_norm=True, n_kpts=4):
        super(PropNet, self).__init__()
        # Node Encoder (Linear)
        self.node_encoder = MLP(node_dim_in, hidden_size, batchnorm=batch_norm, n_features=n_kpts)

        # Edge Encoder (Linear), one per edge type
        self.edge_encoders = nn.ModuleList(
            [MLP(node_dim_in * 2 + edge_dim_in, hidden_size, batchnorm=batch_norm, n_features=n_kpts ** 2) for _ in
             range(edge_type_num)])

        # node propagator
        self.node_propagator = MLP(hidden_size * 3, hidden_size, n_layers=1, batchnorm=batch_norm, n_features=n_kpts)

        # edge propagator
        self.edge_propagators = nn.ModuleList()
        for i in range(pstep):
            edge_propagator = nn.ModuleList([
                MLP(hidden_size * 3, hidden_size, n_layers=1, batchnorm=batch_norm, n_features=n_kpts ** 2) for _ in
                range(edge_type_num)
            ])
            self.edge_propagators.append(edge_propagator)

        # node predictor
        self.node_predictor = nn.Sequential(MLP(2 * hidden_size, hidden_size, batchnorm=batch_norm, n_features=n_kpts),
                                            nn.Linear(hidden_size, node_dim_out))

        # edge predictor
        self.edge_predictor = nn.Sequential(
            MLP(hidden_size * 2, hidden_size, batchnorm=batch_norm, n_features=n_kpts ** 2),
            nn.Linear(hidden_size, edge_dim_out))

    def forward(self, nodes, edges=None, edge_type=None, ignore_node=False, ignore_edge=False):
        B, K, S = nodes.shape

        encoding_node = self.node_encoder(nodes)

        # edge_enc
        node_rep_r = nodes.unsqueeze(2).repeat(1, 1, K, 1)
        node_rep_s = nodes.unsqueeze(1).repeat(1, K, 1, 1)
        if edges is not None:
            tmp = torch.cat([node_rep_r, node_rep_s, edges], dim=-1)
        else:
            tmp = torch.cat([node_rep_r, node_rep_s], dim=-1)

        tmp = tmp.view(B, K * K, -1)
        edge_encs = [f(tmp).view(B, K, K, -1) for f in self.edge_encoders]
        edge_enc = torch.stack(edge_encs, dim=3)

        if edge_type is not None:
            edge_enc = edge_enc * edge_type.view(B, K, K, -1, 1)
        edge_enc = edge_enc.sum(3)

        node_effect = encoding_node
        edge_effect = edge_enc
        for module_list in self.edge_propagators:
            # calculate edge_effect
            node_effect_r = node_effect.unsqueeze(2).repeat(1, 1, K, 1)
            node_effect_s = node_effect.unsqueeze(1).repeat(1, K, 1, 1)
            tmp = torch.cat([node_effect_r, node_effect_s, edge_effect], 3)
            tmp = tmp.view(B, K * K, -1)

            edge_effects = [f(tmp).view(B, K, K, 1, -1) for f in module_list]
            # edge_effect: B x K x K x edge_type_num x nf
            edge_effect = torch.cat(edge_effects, dim=3)

            if edge_type is not None:
                edge_effect = edge_effect * edge_type.view(B, K, K, -1, 1)
            edge_effect = edge_effect.sum(3)
            edge_effect_agg = edge_effect.sum(2)

            tmp = torch.cat([encoding_node, node_effect, edge_effect_agg], 2)
            node_effect = self.node_propagator(tmp)

        node_effect = torch.cat([node_effect, encoding_node], dim=2)
        edge_effect = torch.cat([edge_effect, edge_enc], dim=3).view(B, K * K, -1)

        if ignore_node:
            edge_pred = self.edge_predictor(edge_effect)
            return edge_pred.view(B, K, K, -1)
        if ignore_edge:
            node_pred = self.node_predictor(node_effect)
            return node_pred

        node_pred = self.node_predictor(node_effect)
        edge_pred = self.edge_predictor(edge_effect).view(B, K, K, -1)

        return node_pred, edge_pred


if __name__ == '__main__':
    kp_ab = torch.randn(2, 32, 4, 2)
    kp_cd = torch.randn(2, 4, 4, 6)
    model = DynaNetGNN(4)
    graph = model.graph_inference(kp_ab)
    dyn = model.dynam_prediction(kp_cd, graph)
