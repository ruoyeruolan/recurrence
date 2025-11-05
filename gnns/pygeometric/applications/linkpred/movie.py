# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : movie.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/10 22:04
# @Description:

from typing import Tuple
import pandas as pd
import os.path as osp

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import download_url, extract_zip, HeteroData

from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from utils.visiualization import plot_roc_curve
from models.model import HeteroLinkPredictor
# from utils import hetero2networkx, draw_hetero

# 同质图: 节点和边的类型只有一种
# 异质图: 节点和边至少一个类型不唯一

# "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def create_hetero():
    if not osp.exists("./data/movie/ml-latest-small/movies.csv"):
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        extract_zip(download_url(url, "."), "./data/movie")

    movies_path = "./data/movie/ml-latest-small/movies.csv"
    ratings_path = "./data/movie/ml-latest-small/ratings.csv"

    movie = pd.read_csv(movies_path, index_col=0)
    rating = pd.read_csv(ratings_path)

    movie_x = torch.from_numpy(movie["genres"].str.get_dummies("|").values).float()

    userId = rating["userId"].unique()
    user_idx = torch.from_numpy(
        pd.merge(
            rating["userId"],
            pd.DataFrame({"userId": userId, "mappedId": pd.RangeIndex(len(userId))}),
            left_on="userId",
            right_on="userId",
            how="left",
        )["mappedId"].values
    ).long()

    movie_idx = torch.from_numpy(
        pd.merge(
            rating["movieId"],
            pd.DataFrame(
                {"movieId": movie.index, "mappedId": pd.RangeIndex(movie.shape[0])}
            ),
            left_on="movieId",
            right_on="movieId",
            how="left",
        )["mappedId"].values
    ).long()

    # edge_index: [2, ncol]
    edge_index = torch.stack([user_idx, movie_idx], dim=0)

    data = HeteroData()
    data["user"].node_id = torch.arange(len(userId))
    data["movie"].node_id = torch.arange(movie.shape[0])
    data["movie"].x = movie_x
    data["user", "rates", "movie"].edge_index = edge_index
    data["movie", "rev_rates", "user"].edge_index = edge_index.flip(0)

    transforms = RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        add_negative_train_samples=True,
        neg_sampling_ratio=1,
        edge_types=data.edge_types[0],
        rev_edge_types=data.edge_types[1],
    )

    train_data, val_data, test_data = transforms(data)

    # data.node_types
    # data.edge_types
    # data.edge_index_dict

    # G, node_color_list, color_map, sampled_nodes = hetero2networkx(
    #     data, max_nodes_per_type=100
    # )
    # draw_hetero(data, G, node_color_list, color_map, sampled_nodes, figsize=(16, 9))
    return data, train_data, val_data, test_data


def get_loader(data: HeteroData, edge_type, batch_size: int = 128):
    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[30] * 2,
        edge_label=data[edge_type].edge_label,
        edge_label_index=(edge_type, data[edge_type].edge_label_index),
        batch_size=batch_size,
        shuffle=True,
    )
    return loader


def model_train(
    model: Module,
    loader: LinkNeighborLoader,
    edge_type: Tuple[str, str, str],
    optimizer: torch.optim.Optimizer,
    device: str | None = None,
):
    model.train()

    model.to(device)
    total_loss = total_links = 0
    for data in tqdm(loader, desc="Iteration"):
        optimizer.zero_grad()
        data = data.to(device)
        label = data[edge_type].edge_label
        logits: Tensor = model(data, edge_type)
        loss: Tensor = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * logits.numel()
        total_links += logits.numel()
    return total_loss / total_links


def evaluate(
    model: Module,
    loader: LinkNeighborLoader,
    edge_type: Tuple[str, str, str],
    device: str | None = None,
):
    model.eval()

    model.to(device)
    labels = []
    preds = []
    for data in tqdm(loader, desc="Iteration"):
        with torch.no_grad():
            data = data.to(device)
            label = data[edge_type].edge_label
            logits = model(data, edge_type)
            pred = torch.sigmoid(logits)

            preds.append(pred)
            labels.append(label)

    preds = torch.cat(preds, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    auc = roc_auc_score(labels, preds)
    return auc, preds, labels


def main():
    data, train_data, val_data, test_data = create_hetero()
    edge_type = data.edge_types[0]
    train_loader = get_loader(train_data, edge_type)
    val_loader = get_loader(val_data, edge_type)
    test_loader = get_loader(test_data, edge_type)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = HeteroLinkPredictor(data)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    for epoch in range(1, 11):
        loss = model_train(model, train_loader, edge_type, optimizer, device)
        val_acc, _, _ = evaluate(model, val_loader, edge_type, device)
        print(f"Epoch: {epoch}, Loss: {loss: .4f}, Val Acc: {val_acc: .4f}")

    test_auc, test_preds, test_labels = evaluate(model, test_loader, edge_type, device)
    print(f"Test AUC: {test_auc:.4f}")

    plot_roc_curve(test_labels, test_preds, title="Test Set ROC Curve")
