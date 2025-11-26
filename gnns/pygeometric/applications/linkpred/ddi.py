# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : ddi.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/11/14 22:49
# @Description:

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Module
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv


def preprocess():
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root="./data/ogb")
    data = dataset[0]

    data.node_id = torch.arange(data.num_nodes)  # type: ignore

    spliter = RandomLinkSplit(
        num_val=0.1, num_test=0.2, add_negative_train_samples=True
    )

    train_data, val_data, test_data = spliter(data=data)
    return data, train_data, val_data, test_data


def get_loader(data):
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[30] * 2,
        edge_label=data.edge_label,
        edge_label_index=data.edge_label_index,
        batch_size=128,
    )
    return loader


class SAGE(Module):
    def __init__(self, num_nodes: int, hidden: int) -> None:
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=num_nodes, embedding_dim=hidden)

        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)

        # self.predictor = None

    def forward(self, node_id: Tensor, edge_index: Tensor):
        x: Tensor = self.embeddings(node_id)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)

        return self.conv3(x, edge_index)
