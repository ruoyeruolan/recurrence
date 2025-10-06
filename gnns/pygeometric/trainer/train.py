# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : evaluation.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/05 22:17
# @Description:

import torch

from tqdm.auto import tqdm

from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch_geometric.loader import DataLoader


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
