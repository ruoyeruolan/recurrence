# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : models.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/04 21:20
# @Description:

from typing import Optional

import torch.nn.functional as F

from torch import Tensor
from torch.nn import Module, ModuleList, BatchNorm1d, Linear

from torch_geometric.nn.pool import ASAPooling

from torch_geometric.nn import GCNConv, global_mean_pool


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
