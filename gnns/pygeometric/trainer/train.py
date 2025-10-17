# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : evaluation.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/05 22:17
# @Description:

import torch

from tqdm.auto import tqdm
from typing import Tuple

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch_geometric.loader import DataLoader

from torch_geometric.data import HeteroData


def train(
    model: Module,
    loader: DataLoader,
    criterion: _Loss,
    optimizer: Optimizer,
):
    model.train()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model.to(device)

    ngraphs = total_loss = 0
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_weight, data.batch)
        loss = criterion(pred, data.y)
        loss.backward()
        optimizer.step()

        ngraphs += data.num_graphs
        total_loss += loss.item() * data.num_graphs
    return total_loss / ngraphs if ngraphs > 0 else 0.0


def train_hetero(
    model: Module,
    data: HeteroData,
    edge_type: Tuple[str, str, str],
    decoder,
    optimizer: Optimizer,
    criterion: _Loss,
):
    model.train()
    src, _, dst = edge_type
    labels = data[edge_type].edge_label
    optimizer.zero_grad()
    z_dict = model(data)
    logits = decoder(z_dict[src], z_dict[dst], data.edge_label_index)
    loss: Tensor = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
