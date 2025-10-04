# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : create_data.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/01 13:43
# @Description:

import pandas as pd
import os.path as osp

import networkx as nx

from collections import defaultdict
from typing import List, Callable, Optional, Tuple

import torch
from torch_geometric.io import fs
import torch_geometric.utils as tu
from torch_geometric.data import Data, InMemoryDataset, download_url


class BiologyDataset(InMemoryDataset):
    urls = defaultdict(list)
    urls["Alzheimer"] = [
        "https://raw.githubusercontent.com/sdos1/cs224w_adni_files/main/final_diagnosis.csv",
        "https://raw.githubusercontent.com/sdos1/cs224w_adni_files/main/protein_adjacency_matrix.csv",
        "https://raw.githubusercontent.com/sdos1/cs224w_adni_files/main/log_transformed_ADNI_expression_data_with_covariates.csv",
    ]

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        # use_node_atrr: bool = False,
        # use_edge_atrr: bool = False,
    ) -> None:
        self.name = name
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        data, self.slices, self.sizes, data_cls = fs.torch_load(self.processed_paths[0])

        self.data = data_cls.from_dict(data)

    @property
    def get_urls(self) -> List[str]:
        return self.urls.get(self.name, [])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def raw_file_names(self) -> List[str]:
        return [name.split("/")[-1] for name in self.get_urls]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        for url in self.get_urls:
            download_url(url, self.raw_dir)

    def read_alzheimer_data(self) -> Tuple[pd.DataFrame, ...]:
        diagnosis = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[0]}")
        adj = pd.read_csv(
            f"{self.raw_dir}/{self.raw_file_names[1]}",
            usecols=range(1, 52),
            skiprows=1,
            header=None,
            names=range(51),
        )

        expression = pd.read_csv(
            f"{self.raw_dir}/{self.raw_file_names[2]}",
            skiprows=1,
            header=None,
            usecols=range(16, 67),
            names=range(51),
        )
        return diagnosis, adj, expression

    def process(self):
        diagnosis, adj, expression = self.read_alzheimer_data()
        diagnosis = diagnosis.assign(
            encoder=diagnosis["final_diagnosis"].apply(lambda x: 0 if x == "AD" else 1)
        )

        G: nx.Graph = nx.from_pandas_adjacency(adj)
        G_convert: Data = tu.from_networkx(G)
        x_tensor = (
            torch.from_numpy(expression.to_numpy()).unsqueeze(dim=1).float()
        ).transpose(1, 2)  # torch.Size([565, 51, 1])
        diagnosis_tensor = torch.from_numpy(diagnosis["encoder"].to_numpy()).long()
        positional_encoder = (
            torch.rand(51, 3).float().unsqueeze(dim=0).expand(x_tensor.shape[0], 51, 3)
        )  # torch.Size([565, 51, 3])
        x = torch.cat((x_tensor, positional_encoder), dim=2)

        data_list = [
            Data(
                x=x[i],
                edge_index=G_convert.edge_index,
                edge_attr=G_convert.weight,
                y=diagnosis_tensor[i],
            )
            for i in range(x.shape[0])
        ]

        self.data, self.slices = self.collate(data_list)

        assert isinstance(self._data, Data)
        fs.torch_save(
            (self._data.to_dict(), self.slices, len(data_list), self._data.__class__),
            self.processed_paths[0],
        )


def demo():
    dataset = BiologyDataset(
        root="pygeometric/data/biology", name="Alz", force_reload=True
    )
    dataset[0]
