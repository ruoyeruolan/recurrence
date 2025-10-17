# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : movie.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/10 22:04
# @Description:

import pandas as pd
import os.path as osp
from typing import Tuple

import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import download_url, extract_zip, HeteroData

from utils import hetero2networkx, draw_hetero

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

    movie_x = torch.from_numpy(movie["genres"].str.get_dummies("|").values).long()

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
