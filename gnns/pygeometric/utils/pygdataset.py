# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : create_data.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/02 19:35
# @Description:

import torch
import pandas as pd
import os.path as osp

import networkx as nx

import torch_geometric.utils as tu
from torch_geometric.io import fs

from collections import defaultdict
from typing import Callable, List, Tuple
from torch_geometric.data import InMemoryDataset, download_url, Data


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
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)

        data, self.slices, _, cls = fs.torch_load(self.processed_paths[0])
        self.data = cls.from_dict(data)

    @property
    def raw_dir(self) -> str:  # root/Alzheimer/raw,processed
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return [file.split("/")[-1] for file in self.get_url]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    @property
    def get_url(self) -> List[str]:
        return self.urls[self.name]

    def download(self):
        for url in self.get_url:
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

    def process(self) -> None:
        diagnosis, adj, expression = self.read_alzheimer_data()
        diagnosis = diagnosis.assign(
            encoder=diagnosis["final_diagnosis"].apply(lambda x: 0 if x == "AD" else 1)
        )

        G = nx.from_pandas_adjacency(adj)
        G_convert = tu.from_networkx(G)

        x_tensor = (
            torch.from_numpy(expression.to_numpy()).float().unsqueeze(dim=2)
        )  # torch.Size([565, 51, 1])
        diagnosis_tensor = torch.from_numpy(diagnosis["encoder"].to_numpy()).long()

        torch.manual_seed(42)
        positional_encoder = (
            torch.rand((51, 3)).float().expand(x_tensor.shape[0], 51, 3)
        )  # torch.Size([565, 51, 3])

        x = torch.cat((x_tensor, positional_encoder), dim=2)

        # nx.draw(G)
        # plt.show()

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

    def __repr__(self) -> str:
        return f"{self.name}({len(self)})"


def demo():
    dataset = BiologyDataset(
        root="./data/biology/", name="Alzheimer", force_reload=True
    )
    dataset.raw_dir

    dataset[0]

    diagnosis, adj, expression = dataset.read_alzheimer_data()
    diagnosis = diagnosis.assign(
        encoder=diagnosis["final_diagnosis"].apply(lambda x: 0 if x == "AD" else 1)
    )

    diagnosis["encoder"].value_counts()
