# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : cora.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/09/23 20:54
# @Description:


import torch
import torch.nn.functional as F

from torch import Tensor

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(input=x, p=0.5, training=self.training)

        return self.conv2(x, edge_index)


def model_train(model: torch.nn.Module, data, optimizer: torch.optim.Optimizer):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    loss = torch.nn.CrossEntropyLoss()(pred[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def model_test(model: torch.nn.Module, data):
    model.eval()

    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)

        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
    return acc


def main():
    # dirs = "/Users/wakala/IdeaProjects/Projects/recurrence/gnns/pygeometric/data"
    dataset = Planetoid(root="./data/planetoid", name="Cora")
    data = dataset[0]

    model = GCN(dataset.num_edge_features, dataset.num_classes)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 101):
        loss = model_train(model, data, optimizer)
        print(f"Epoch: {epoch}, Loss: {loss: .4f}")

        if epoch % 10 == 0:
            acc = model_test(model, data)
            print(f"Epoch: {epoch}, Acc: {acc: .4f}")
