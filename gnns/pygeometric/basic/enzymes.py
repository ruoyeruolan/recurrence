# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : enzymes.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/01 21:36
# @Description:

import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_mean_pool

from utils import draw_random_graph_samples

from sklearn.model_selection import train_test_split

from basic.mutag import model_train


class GINClassification(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
        )

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
        )

        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )
        )

        self.fc = nn.Linear(32, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch=batch)
        classification = self.fc(x)
        return classification


def preprocess():
    dataset = TUDataset(root="./data/tudatasets", name="ENZYMES", use_node_attr=True)
    dataset.shuffle()

    train, test = train_test_split(dataset, test_size=0.2, random_state=999)

    train_loader = DataLoader(train, batch_size=32)
    test_loader = DataLoader(test, batch_size=32)
    return dataset, train_loader, test_loader


def model_test(model: GINClassification, loader: DataLoader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for data in loader:
            pred, _ = model(data.x, data.edge_index, data.batch)
            pred = pred.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs
    return correct / total


def main():
    dataset, train_loader, test_loader = preprocess()

    draw_random_graph_samples(dataset, name="ENZYMES")

    model = GINClassification(dataset.num_node_features, dataset.num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    for epoch in range(1, 101):
        optimizer.zero_grad()
        loss = model_train(model, train_loader, optimizer, criterion)
        print(f"Epoch: {epoch}, Loss: {loss: .4f}")

    model_test(model, test_loader)
