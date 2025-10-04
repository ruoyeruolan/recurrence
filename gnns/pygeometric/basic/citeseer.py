# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : citeseer.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/09/26 23:27
# @Description:

import torch

from torch import Tensor
from torch.nn import Module

import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.transforms import NormalizeFeatures

from basic import model_train


class GCN(Module):
    def __init__(self, in_channels: int, out_channels: int, layer: str = "GCN") -> None:
        super().__init__()

        dit = {
            "GCN": GCNConv,
            "GAT": GATConv,
        }
        self.conv1 = dit[layer](in_channels, 16)
        self.conv2 = dit[layer](16, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        return self.conv2(x, edge_index)


def model_test(model: Module, data):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = (pred[mask] == data.y[mask]).sum()
            acc = int(correct) / int(mask.sum())
            accs.append(acc)
        print(
            f"Train Acc: {accs[0]: .4f}, Val Acc: {accs[1]: .4f}, Test Acc: {accs[2]: .4f}"
        )


def main():
    dataset = Planetoid(
        root="./data/planetoid", name="CiteSeer", transform=NormalizeFeatures()
    )
    data = dataset[0]

    model = GCN(dataset.num_features, dataset.num_classes, layer="GAT")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 101):
        loss = model_train(model, data, optimizer)
        print(f"Epoch: {epoch}, Loss: {loss: .4f}")

    model_test(model, data)
