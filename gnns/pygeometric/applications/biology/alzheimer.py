# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : alzheimer.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/05 21:13
# @Description:


import copy
import torch

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from trainer.train import train
from models.model import GCNGraph
from evaluation.evaluate import evaluate
from utils.pygdataset import BiologyDataset
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


def preprocess():
    dataset = BiologyDataset("./data/biology/", name="Alzheimer")

    labels = [data.y.item() for data in dataset]
    label_counts = torch.bincount(torch.tensor(labels))
    print(f"Label distribution: {label_counts}")
    print(f"Class 0: {label_counts[0].item()}, Class 1: {label_counts[1].item()}")
    print(f"Imbalance ratio: {label_counts[1].item() / label_counts[0].item():.2f}")

    weights = 1.0 / label_counts.float()
    weights = weights / weights.sum()

    dataset.shuffle()

    train, val, test = random_split(
        dataset,
        lengths=[
            int(len(dataset) * 0.6),
            int(len(dataset) * 0.2),
            int(len(dataset) * 0.2),
        ],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train, batch_size=32)  # type: ignore
    val_loader = DataLoader(val, batch_size=32)  # type: ignore
    test_loader = DataLoader(test, batch_size=32)  # type: ignore
    return dataset, train_loader, val_loader, test_loader, weights


def main():
    dataset, train_loader, val_loader, test_loader, weights = preprocess()

    model = GCNGraph(
        dataset.num_features, 256, dataset.num_classes, nlayers=5, dropout=0.5
    )
    model.reset_parameters()
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to("cuda"))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    best_val_acc = 0.0
    for epoch in range(1, 101):
        loss = train(model, train_loader, criterion, optimizer)

        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        test_acc = evaluate(model, test_loader)
        print(
            f"Epoch: {epoch}, Loss: {loss: .4f}, Train Acc: {train_acc: .4f}, Val Acc: {val_acc: .4f}, Test Acc: {test_acc: .4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

    train_acc = evaluate(best_model, train_loader)
    val_acc = evaluate(best_model, val_loader)
    test_acc = evaluate(best_model, test_loader)
    print(
        f"Train Acc: {train_acc: .4f}, Val Acc: {val_acc: .4f}, Test Acc: {test_acc: .4f}"
    )


# dataset, train_loader, val_loader, test_loade, weights = preprocess()

# model = GCNGraph(
#     dataset.num_features, 256, dataset.num_classes, nlayers=5, dropout=0.5
# )
# criterion = torch.nn.CrossEntropyLoss(weight=weights.to('cuda'))
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

# for epoch in range(1, 101):
#     loss = train(model, train_loader, criterion, optimizer)

#     # 检查梯度
#     total_norm = 0
#     for name, p in model.named_parameters():
#         if p.grad is not None:
#             param_norm = p.grad.norm().item()
#             total_norm += param_norm ** 2
#             if epoch == 1:  # 第一个 epoch 打印详细信息
#                 print(f"{name}: grad norm = {param_norm:.4f}")
#     total_norm = total_norm ** 0.5

#     train_acc = evaluate(model, train_loader)
#     val_acc = evaluate(model, val_loader)
#     test_acc = evaluate(model, test_loader)

#     print(
#         f"Epoch: {epoch}, Loss: {loss:.4f}, Grad norm: {total_norm:.4f}, "
#         f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
#     )
