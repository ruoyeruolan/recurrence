# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : rellink.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/07 21:28
# @Description:

import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets import RelLinkPredDataset, MovieLens100K

dataset = RelLinkPredDataset(root="./data/rellink", name="FB15k-237")
dataset[0]

dataset = MovieLens100K(root="./data/movie")
dataset[0]

isinstance(dataset[0], HeteroData)
