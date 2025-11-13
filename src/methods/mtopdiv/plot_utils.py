from itertools import chain
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from .utils import transform_attention_scores_to_distances


def plot_subgraph_nodes(
    G: nx.Graph,
    pos: dict,
    ax: Axes,
    nodes: Iterable,
    color: str = "red",
    node_size: int = 2,
):
    subgraph_nodes = [node for node in G.nodes if node in nodes]
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=subgraph_nodes, node_color=color, node_size=node_size
    )


def spans_to_nodes(
    instruction_span: list[tuple],
    hallucination_span: list[tuple],
    prompt_len: int,
    answer_len: int,
) -> tuple[list]:
    prompt_nodes = [
        range(first[1], second[0])
        for first, second in zip(instruction_span, instruction_span[1:])
    ]
    instruction_nodes = [range(*x) for x in instruction_span]

    hallu_nodes = []
    if not hallucination_span:
        answer_nodes = [range(prompt_len, prompt_len + answer_len)]
    else:
        answer_nodes = []
        t = prompt_len
        for start, end in hallucination_span:
            answer_nodes.append(range(t, start + prompt_len))
            hallu_nodes.append(range(start + prompt_len, end + prompt_len))
            t = end + prompt_len
        answer_nodes.append(range(t, prompt_len + answer_len))

    return prompt_nodes, instruction_nodes, answer_nodes, hallu_nodes


def plot_nodes(*nodes, G: nx.Graph, pos: dict, ax: Axes):
    prompt_nodes, instruction_nodes, answer_nodes, hallu_nodes = nodes

    for prompt, instruct in zip(prompt_nodes, instruction_nodes):
        plot_subgraph_nodes(G, pos, ax, prompt, color="red")
        plot_subgraph_nodes(G, pos, ax, instruct, color="blue")

    plot_subgraph_nodes(G, pos, ax, instruction_nodes[-1], color="blue")

    for answ, hallu in zip(answer_nodes, hallu_nodes):
        plot_subgraph_nodes(G, pos, ax, answ, color="green")
        plot_subgraph_nodes(G, pos, ax, hallu, color="orange")

    plot_subgraph_nodes(G, pos, ax, answer_nodes[-1], color="green")


def plot_mx(
    adj_mx: np.ndarray,
    prompt_len: int,
    instruction_span: list[tuple],
    hallucination_span: list[tuple],
    ax: Axes,
) -> None:
    answer_len = adj_mx.shape[0] - prompt_len
    G = nx.from_numpy_array(adj_mx)

    nodes = spans_to_nodes(instruction_span, hallucination_span, prompt_len, answer_len)

    colors = ["red", "blue", "green", "orange"]
    labels = ["prompt", "instruction", "response", "hallucination"]
    pos = nx.bipartite_layout(G, range(prompt_len))

    plot_nodes(*nodes, G=G, pos=pos, ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=G.edges(chain(*nodes[-2])),
        edge_color="green",
        width=0.5,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=G.edges(chain(*nodes[-1])),
        edge_color="orange",
        width=0.5,
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=label,
        )
        for color, label in zip(colors, labels)
    ]
    plt.legend(handles=handles, loc="best", title="Node colors")


def plot_filtration(
    attn_mx: np.ndarray,
    prompt_len: int,
    instruction_span: list[tuple],
    hallucination_span: list[tuple],
    thresholds: Optional[list] = None,
):
    answer_len = attn_mx.shape[0] - prompt_len
    dist_mx = transform_attention_scores_to_distances(
        attn_mxs=attn_mx[None, ...], zero_out="prompt", len_answer=answer_len
    )[0]
    print(dist_mx.shape)

    if thresholds is None:
        thresholds = (0.005, 0.01, 0.02, 0.05)

    thresholds = [np.quantile(dist_mx[-answer_len, :prompt_len], q) for q in thresholds]

    _, axes = plt.subplots(1, len(thresholds), figsize=(16, 10))
    for i, th in enumerate(thresholds):
        adj_mx = (dist_mx < th).astype(int)
        adj_mx -= np.diag(np.diag(adj_mx))

        adj_mx[-answer_len:, -answer_len:] = 0  # to keep the plot neat
        plot_mx(adj_mx, prompt_len, instruction_span, hallucination_span, axes[i])

    plt.show()


def plot_minimum_tree(
    attn_mx: np.ndarray,
    mtopdiv: float,
    prompt_len: int,
    instruction_span: list[tuple],
    hallucination_span: list[tuple],
    name: str,
    save_path: Path,
):
    plt.figure(figsize=(8, 8))
    ax = None
    answer_len = attn_mx.shape[0] - prompt_len
    dist_mx = transform_attention_scores_to_distances(
        attn_mxs=attn_mx[None, ...], zero_out="prompt", len_answer=answer_len
    )[0]
    dist_mx[:prompt_len, :prompt_len] = 1e-5
    G = nx.from_numpy_array(dist_mx)
    T = nx.minimum_spanning_tree(G)

    nodes = spans_to_nodes(instruction_span, hallucination_span, prompt_len, answer_len)

    colors = ["red", "blue", "green", "orange"]
    labels = ["prompt", "instruction", "response", "hallucination"]
    pos = nx.bipartite_layout(G, range(prompt_len))

    plot_nodes(*nodes, G=G, pos=pos, ax=ax)
    nx.draw_networkx_edges(
        T,
        pos,
        edgelist=T.edges(chain(*nodes[-2])),
        edge_color="green",
        width=[
            1 - elem[-1]["weight"]
            for elem in list(T.edges(chain(*nodes[-2]), data=True))
        ],
    )

    nx.draw_networkx_edges(
        T,
        pos,
        edgelist=T.edges(chain(*nodes[-1])),
        edge_color="orange",
        width=[
            1 - elem[-1]["weight"]
            for elem in list(T.edges(chain(*nodes[-1]), data=True))
        ],
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=label,
        )
        for color, label in zip(colors, labels)
    ]
    plt.legend(handles=handles, loc="best", title="Node colors")
    plt.title(f"Minimum spanning tree: {name}, MTD={mtopdiv:.2f}")
    plt.axis("off")
    plt.savefig(save_path)

    return T
