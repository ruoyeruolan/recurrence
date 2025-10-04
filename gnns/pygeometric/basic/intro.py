# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : intro.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/09/23 15:54
# @Description:

# COO coordinate format
# CSR compressed sparse row format

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

data_ = TUDataset(
    root="/Users/wakala/IdeaProjects/Projects/recurrence/gnns/pygeometric/data/tudatasets",
    name="ENZYMES",
)

data_[0]
