# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : evaluate.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/06 14:13
# @Description:

from tqdm.auto import tqdm

import torch
from torch.nn import Module
from torch_geometric.loader import DataLoader


def evaluate(model: Module, loader: DataLoader):
    model.eval()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model.to(device)

    with torch.no_grad():
        correct = ngrphas = 0
        for data in tqdm(loader, desc="Evaluting"):
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_weight, data.batch).argmax(
                dim=1
            )
            correct += (pred == data.y).sum()
            ngrphas += data.num_graphs
    return int(correct) / ngrphas if ngrphas > 0 else 0.0
