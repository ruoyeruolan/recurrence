# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : utils.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/09/27 19:50
# @Description:

import os
import random
import networkx as nx
import matplotlib.pyplot as plt

from typing import List, Optional

import pandas as pd
from matplotlib.lines import Line2D


def draw_random_graph_samples(
    dataset,
    # num_classes,
    num_samples=20,
    name: str = "",
    num_rows: int = 4,
    num_cols: int = 5,
):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    fig.suptitle(f"Random Samples of {name} Graphs")

    cmap = plt.get_cmap("tab10", dataset.num_classes)

    random_indices = random.sample(range(len(dataset)), num_samples)

    for i, index in enumerate(random_indices):
        row = i // num_cols
        col = i % num_cols

        data = dataset[index]
        edge_index, y = data.edge_index, data.y
        G = nx.Graph()
        for src, dst in edge_index.t().tolist():
            G.add_edge(src, dst)

        # Map class labels to color

        pos = nx.spring_layout(G)  # Positions for all nodes
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=cmap(data.y),
            node_size=100,
            edge_color="k",
            linewidths=1,
            font_size=5,
            ax=axs[row, col],  # type: ignore
        )
        axs[row, col].set_title(f"Graph Sample {index}")  # type: ignore

    handles = [
        Line2D(
            [],
            [],
            color=cmap(i),
            marker="o",
            linestyle="",
            markersize=10,
            label=f"Class {i}",
        )
        for i in range(dataset.num_classes)
    ]
    plt.legend(
        handles=handles,
        title="Class Labels",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plt.show(block=False)
