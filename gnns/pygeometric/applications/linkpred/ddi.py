# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : ogblddi.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/22 21:29
# @Description:

import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit, ToSparseTensor
from torch_geometric.nn import SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset
from tqdm.auto import tqdm


def get_loader(data, num_neighbors=[30] * 2, batch_size: int = 128):
    row, col, _ = data.adj_t.t().coo()
    data.edge_index = torch.stack([row, col], dim=0).contiguous()

    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        edge_label=data.edge_label,
        edge_label_index=data.edge_label_index,
        batch_size=batch_size,
    )
    return loader


def preprocess():
    dataset = PygLinkPropPredDataset(
        "ogbl-ddi", "./data/ogb", transform=ToSparseTensor()
    )
    data: Data = dataset[0]  # type: ignore
    if getattr(data, "x", None) is None:
        data.x = torch.eye(data.num_nodes, dtype=torch.float)  # type: ignore
    # data.adj_t
    # to_dense_adj(data.adj_t)
    # G = convert.to_networkx(data, to_undirected=True)
    spliter = RandomLinkSplit(num_test=0.2, num_val=0.1, is_undirected=True)

    train_data, val_data, test_data = spliter(data)
    # train_data.edge_index = train_data.adj_t.t().coo()
    return data, train_data, val_data, test_data


class SAGE(Module):
    def __init__(self, in_channels: int, hidden: int = 64) -> None:
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)

        self.logits = DotProductLinkPredictor()

    def forward(self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)
        return self.logits(x, edge_label_index)


class DotProductLinkPredictor(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, edge_label_index):
        src = x[edge_label_index[0]]
        dst = x[edge_label_index[1]]
        return (src * dst).sum(dim=-1)


def model_train(
    model: Module,
    loader: LinkNeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
):
    model.train()

    total_loss = total_links = 0
    for data in tqdm(loader, desc="Train"):
        data = data.to(device)
        optimizer.zero_grad()
        logits: torch.Tensor = model(data.x, data.edge_index, data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, data.edge_label)
        loss.backward()
        optimizer.step()

        total_links += logits.numel()
        total_loss += loss.item() * logits.numel()
    return total_loss / total_links


def main():
    data, train_data, val_data, tets_data = preprocess()
    train_loader = get_loader(train_data)
    # val_loader = get_loader(val_data)
    # test_loader = get_loader(tets_data)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = SAGE(data.num_features).to(device)
    optimier = torch.optim.Adam(params=model.parameters(), lr=0.001)

    for epoch in range(1, 11):
        loss = model_train(model, train_loader, optimier, device=device)
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")
