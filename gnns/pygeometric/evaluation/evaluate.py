# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : evaluate.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/06 14:13
# @Description:

from tqdm.auto import tqdm
from typing import Tuple

import torch
from torch.nn import Module
from torch_geometric.loader import DataLoader

from torch_geometric.data import HeteroData

from sklearn.metrics import roc_auc_score


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


def evaluate_hetero(
    model: Module,
    data: HeteroData,
    edge_type: Tuple[str, str, str],
    decoder,
):
    model.eval()

    src, _, dst = edge_type
    z_dict = model(data)
    logits = decoder(z_dict[src], z_dict[dst], data.edge_label_index)
    probs = torch.nn.Sigmoid()(logits)

    pred = (probs > 0.5).float()
    labels = data[edge_type].edge_label

    acc = (pred == labels).mean()
    auc = roc_auc_score(labels, pred)
    return acc, auc
