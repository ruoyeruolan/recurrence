# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : models.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/04 21:20
# @Description:

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Module, ModuleList, BatchNorm1d, Linear, ModuleDict, Embedding

from torch_geometric.nn.pool import ASAPooling

from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    SAGEConv,
    HeteroConv,
    to_hetero,
)


class GCN(Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        nlayers: int,
        dropout: float = 0.5,
        embedding: bool = True,
    ) -> None:
        super().__init__()

        self.nlayers = nlayers
        self.dropout = dropout
        self.embedding = embedding

        self.convs = ModuleList()
        self.batchNormals = ModuleList()

        for n in range(nlayers):
            if n == 0:
                self.convs.append(GCNConv(in_channels, hidden))
            else:
                self.convs.append(GCNConv(hidden, hidden))

            self.batchNormals.append(BatchNorm1d(hidden))

        self.lin = Linear(hidden, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.batchNormals:
            bn.reset_parameters()

        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor]):
        for n in range(self.nlayers):
            x = self.convs[n](x, edge_index, edge_weight)
            x = self.batchNormals[n](x)
            x = x.relu()

            if n < self.nlayers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.embedding:
            return x
        else:
            return self.lin(x)


class GCNGraph(Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        nlayers: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.gnn_node1 = GCN(
            in_channels, hidden, hidden, nlayers, dropout, embedding=True
        )
        self.gnn_node2 = GCN(hidden, hidden, hidden, nlayers, dropout, embedding=True)
        self.gnn_node3 = GCN(hidden, hidden, hidden, nlayers, dropout, embedding=True)

        self.asap = ASAPooling(hidden, dropout=dropout, add_self_loops=False)
        # self.topk1 = TopKPooling(hidden)
        # self.topk2 = TopKPooling(hidden)

        self.lin = Linear(hidden, out_channels)

    def reset_parameters(self):
        self.gnn_node1.reset_parameters()
        self.gnn_node2.reset_parameters()
        self.gnn_node3.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        batch: Tensor,
    ):
        x = self.gnn_node1(x, edge_index, edge_weight)
        (
            x,
            edge_index,
            edge_weight,
            batch,
            _,
        ) = self.asap(x, edge_index, edge_weight, batch)

        x = self.gnn_node2(x, edge_index, edge_weight)
        x, edge_index, edge_weight, batch, _ = self.asap(
            x, edge_index, edge_weight, batch
        )

        x = self.gnn_node3(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch=batch)

        return self.lin(x)


class HeteroLinkPredictor_(Module):
    def __init__(self, data: HeteroData, hidden: int, nlayers: int = 3) -> None:
        super().__init__()

        self.node_types, self.edge_types = data.node_types, data.edge_types

        self.embedds = ModuleDict()
        for ntype in self.node_types:
            if data[ntype].num_features == 0:
                self.embedds[ntype] = Embedding(data[ntype].num_nodes, 32)

        self.convs = ModuleList()
        for _ in range(nlayers):
            conv = HeteroConv(
                {edge_type: SAGEConv((-1, -1), hidden) for edge_type in self.edge_types}
            )
            self.convs.append(conv)

        self.lin = ModuleDict()
        for ntype in self.node_types:
            self.lin[ntype] = Linear(hidden, hidden)

    def forward(self, data: HeteroData):
        x_dict: dict = {}

        for ntyep in self.node_types:
            if ntyep in data.x_dict and data.x_dict[ntyep] is not None:
                x_dict = data.x_dict[ntyep]
            elif ntyep in self.embedds:
                idx = torch.arange(data[ntyep].num_nodes)
                x_dict = self.embedds[ntyep](idx)
            else:
                pass

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: val.relu() for key, val in x_dict.items()}

        z_dict = {key: self.lin[key](val) for key, val in x_dict.items()}
        return z_dict


class GNN(Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()

        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        return x


class Decoder(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        x_user: torch.Tensor,
        x_movie: torch.Tensor,
        edge_label_index: torch.Tensor,
    ):
        x_user_feat = x_user[edge_label_index[0]]
        x_movie_feat = x_movie[edge_label_index[1]]

        return (x_user_feat * x_movie_feat).sum(dim=-1)


class HeteroLinkPredictor(torch.nn.Module):
    def __init__(self, data: HeteroData, hidden: int = 64) -> None:
        super().__init__()

        self.movie_lin = torch.nn.Linear(data["movie"].num_features, hidden)
        self.user_embed = torch.nn.Embedding(data["user"].num_nodes, hidden)
        self.movie_embed = torch.nn.Embedding(data["movie"].num_nodes, hidden)

        self.gnn = to_hetero(GNN(hidden), data.metadata())

        self.classification = Decoder()

    def forward(self, data: HeteroData, edge_type: Tuple[str, str, str]):
        x_dict: dict = {
            "user": self.user_embed(data["user"].node_id),
            "movie": self.movie_lin(data["movie"].x)
            + self.movie_embed(data["movie"].node_id),
        }

        x_dict: dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classification(
            x_dict["user"], x_dict["movie"], data[edge_type].edge_label_index
        )
        return pred
