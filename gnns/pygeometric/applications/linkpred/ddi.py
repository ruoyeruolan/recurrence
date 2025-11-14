# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : ddi.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/11/14 22:49
# @Description:

import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.transforms import RandomLinkSplit


def preprocess():
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root="./data/ogb")
    data = dataset[0]

    data.node_id = torch.arange(data.num_nodes)  # type: ignore

    spliter = RandomLinkSplit(
        num_val=0.1, num_test=0.2, add_negative_train_samples=True
    )

    train_data, val_data, test_data = spliter(data=data)
    return data, train_data, val_data, test_data
