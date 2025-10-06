# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : mutag.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/09/28 00:04
# @Description:

import torch
from torch import Tensor

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_add_pool

from sklearn.model_selection import train_test_split

from utils.visiualization import draw_random_graph_samples


class GCN(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, layer: str = "Graph"
    ) -> None:
        super().__init__()

        dit = {"GCN": GCNConv, "GAT": GATConv, "Graph": GraphConv}

        self.conv1: MessagePassing = dit[layer](in_channels, 16)
        self.conv2: MessagePassing = dit[layer](16, 32)
        self.conv3: MessagePassing = dit[layer](32, 32)

        self.lin = torch.nn.Linear(32, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = x.relu()

        x = self.conv3(x, edge_index)
        x = x.relu()

        x = global_add_pool(x, batch=batch)
        return self.lin(x)


def preprocess():
    dirs = "pygeometric/data/tudatasets"
    dataset = TUDataset(root=dirs, name="MUTAG")
    # dataset.shuffle()

    train, test = train_test_split(
        dataset, test_size=0.2, shuffle=True, random_state=999
    )

    train_loader = DataLoader(train, batch_size=16)
    test_loader = DataLoader(test, batch_size=16)
    return dataset, train_loader, test_loader


def model_train(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.CrossEntropyLoss,
):
    model.train()
    ngraphs = 0
    loss_ = 0
    for data in loader:
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.batch)
        loss = criterion(pred, data.y)
        loss.backward()
        optimizer.step()
        ngraphs += data.num_graphs
        loss_ += loss.item() * data.num_graphs
    return loss_ / ngraphs if ngraphs > 0 else 0


def model_test(model: torch.nn.Module, loader: DataLoader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
            correct += (pred == data.y).sum()
            total += data.num_graphs
    return int(correct) / total if total > 0 else 0.0


def main():
    dataset = TUDataset(root="./data/tudatasets", name="MUTAG")
    # dataset.num_classes

    draw_random_graph_samples(dataset, name="MUTAG")

    dataset, train_loader, test_loader = preprocess()
    model = GCN(dataset.num_features, dataset.num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 101):
        loss = model_train(model, train_loader, optimizer, criterion)
        print(f"Epoch: {epoch}, Loss: {loss: .4f}")
    model_test(model, test_loader)
