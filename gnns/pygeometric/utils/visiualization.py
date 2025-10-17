# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : visiualization.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/10/04 20:43
# @Description:

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from torch_geometric.data import HeteroData

from sklearn.metrics import roc_curve, roc_auc_score


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


def hetero2networkx(data: HeteroData, max_nodes_per_type=30):
    random.seed(42)
    np.random.seed(42)

    G = nx.Graph()

    node_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(data.node_types)))
    color_map = {
        node_type: color for node_type, color in zip(data.node_types, node_colors)
    }

    node_info = {}  # {(node_type, original_idx): new_idx}
    node_color_list = []
    node_labels = {}
    current_idx = 0

    sampled_nodes = {}
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        sample_size = min(max_nodes_per_type, num_nodes)
        sampled_indices = np.random.choice(num_nodes, size=sample_size, replace=False)
        sampled_nodes[node_type] = set(sampled_indices.tolist())

        for idx in sampled_indices:
            G.add_node(current_idx)
            node_info[(node_type, idx)] = current_idx
            node_color_list.append(color_map[node_type])
            node_labels[current_idx] = f"{node_type}\n{idx}"
            current_idx += 1

    edge_counts = {}
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        edge_index = data[edge_type].edge_index

        edge_count = 0
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i].item()
            dst_idx = edge_index[1, i].item()

            # 只添加采样节点之间的边
            if (
                src_idx in sampled_nodes[src_type]
                and dst_idx in sampled_nodes[dst_type]
            ):
                src_node = node_info[(src_type, src_idx)]
                dst_node = node_info[(dst_type, dst_idx)]
                G.add_edge(src_node, dst_node)
                edge_count += 1

        edge_counts[f"{src_type}_{relation}_{dst_type}"] = edge_count
    return G, node_color_list, color_map, sampled_nodes


def draw_hetero(data, G, node_color_list, color_map, sampled_nodes, figsize=(4, 3)):
    fig, ax = plt.subplots(figsize=figsize)

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color="gray", ax=ax)

    nx.draw_networkx_nodes(
        G, pos, node_color=node_color_list, node_size=100, alpha=0.8, ax=ax
    )

    legend_elements = []
    for node_type, color in color_map.items():
        num_sampled = len(sampled_nodes[node_type])
        total_num = data[node_type].num_nodes
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=f"{node_type} ({num_sampled}/{total_num})",
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.8, 0.2),
        fontsize=10,
    )

    ax.set_title("Visiualization of HeteroGraph", fontsize=14, fontweight="bold")

    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(labels, probs):
    auc = roc_auc_score(labels, probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc: .4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
