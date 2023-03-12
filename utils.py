import copy
from typing import Dict, List, Optional, Union, Any
from math import sqrt

import torch
from torch import Tensor


from torch_geometric.data.data import Data, warn_or_raise
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.explain.config import ThresholdConfig, ThresholdType
from torch_geometric.typing import EdgeType, NodeType

def visualize_feature_importance(
        explanation,
        path: Optional[str] = None,
        feat_labels: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ):
        r"""Creates a bar plot of the node features importance by summing up
        :attr:`explanation.node_feat_mask` across all nodes.

        Args:
            path (str, optional): The path to where the plot is saved.
                If set to :obj:`None`, will visualize the plot on-the-fly.
                (default: :obj:`None`)
            feat_labels (List[str], optional): Optional labels for features.
                (default :obj:`None`)
            top_k (int, optional): Top k features to plot. If :obj:`None`
                plots all features. (default: :obj:`None`)
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        node_feat_mask = explanation.get('node_feat_mask')
        if node_feat_mask is None:
            raise ValueError(f"The attribute 'node_feat_mask' is not available "
                             f"in '{explanation.__class__.__name__}' "
                             f"(got {explanation.available_explanations})")
        if node_feat_mask.dim() != 2 or node_feat_mask.size(1) <= 1:
            raise ValueError(f"Cannot compute feature importance for "
                             f"object-level 'node_feat_mask' "
                             f"(got shape {node_feat_mask.size()})")

        feat_importance = node_feat_mask.sum(dim=0).cpu().numpy()

        if feat_labels is None:
            feat_labels = range(feat_importance.shape[0])

        if len(feat_labels) != feat_importance.shape[0]:
            raise ValueError(f"The '{explanation.__class__.__name__}' object holds "
                             f"{feat_importance.numel()} features, but "
                             f"only {len(feat_labels)} were passed")

        df = pd.DataFrame({'feat_importance': feat_importance},
                          index=feat_labels)
        df = df.sort_values("feat_importance", ascending=False)
        df = df.round(decimals=3)

        if top_k is not None:
            df = df.head(top_k)
            title = f"Feature importance for top {len(df)} features"
        else:
            title = f"Feature importance for {len(df)} features"

        ax = df.plot(
            kind='barh',
            figsize=(10, 7),
            title=title,
            ylabel='Feature label',
            xlim=[0, float(feat_importance.max()) + 0.3],
            legend=False,
        )
        plt.gca().invert_yaxis()
        ax.bar_label(container=ax.containers[0], label_type='edge')

        if path is not None:
            plt.savefig(path)
        else:
            plt.show()

        plt.close()

def visualize_graph(explanation, path: Optional[str] = None,
                        backend: Optional[str] = None):
        """Visualizes the explanation graph with edge opacity corresponding to
        edge importance.

        Args:
            path (str, optional): The path to where the plot is saved.
                If set to :obj:`None`, will visualize the plot on-the-fly.
                (default: :obj:`None`)
            backend (str, optional): The graph drawing backend to use for
                visualization (:obj:`"graphviz"`, :obj:`"networkx"`).
                If set to :obj:`None`, will use the most appropriate
                visualization backend based on available system packages.
                (default: :obj:`None`)
        """
        edge_mask = explanation.get('edge_mask')
        if edge_mask is None:
            raise ValueError(f"The attribute 'edge_mask' is not available "
                             f"in '{explanation.__class__.__name__}' "
                             f"(got {explanation.available_explanations})")
        visualize_graph_helper(explanation.edge_index, edge_mask, path, backend)

BACKENDS = {'graphviz', 'networkx'}


def has_graphviz() -> bool:
    try:
        import graphviz
    except ImportError:
        return False

    try:
        graphviz.Digraph().pipe()
    except graphviz.backend.ExecutableNotFound:
        return False

    return True


def visualize_graph_helper(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    path: Optional[str] = None,
    backend: Optional[str] = None,
) -> Any:
    r"""Visualizes the graph given via :obj:`edge_index` and (optional)
    :obj:`edge_weight`.
    Args:
        edge_index (torch.Tensor): The edge indices.
        edge_weight (torch.Tensor, optional): The edge weights.
        path (str, optional): The path to where the plot is saved.
            If set to :obj:`None`, will visualize the plot on-the-fly.
            (default: :obj:`None`)
        backend (str, optional): The graph drawing backend to use for
            visualization (:obj:`"graphviz"`, :obj:`"networkx"`).
            If set to :obj:`None`, will use the most appropriate
            visualization backend based on available system packages.
            (default: :obj:`None`)
    """
    if edge_weight is not None:  # Normalize edge weights.
        edge_weight = edge_weight - edge_weight.min()
        edge_weight = edge_weight / edge_weight.max()

    if edge_weight is not None:  # Discard any edges with zero edge weight:
        mask = edge_weight > 1e-7
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))

    if backend is None:
        backend = 'graphviz' if has_graphviz() else 'networkx'

    if backend.lower() == 'networkx':
        return _visualize_graph_via_networkx(edge_index, edge_weight, path)
    elif backend.lower() == 'graphviz':
        return _visualize_graph_via_graphviz(edge_index, edge_weight, path)

    raise ValueError(f"Expected graph drawing backend to be in "
                     f"{BACKENDS} (got '{backend}')")


def _visualize_graph_via_graphviz(
    edge_index: Tensor,
    edge_weight: Tensor,
    path: Optional[str] = None,
) -> Any:
    import graphviz

    suffix = path.split('.')[-1] if path is not None else None
    g = graphviz.Digraph('graph', format=suffix)
    g.attr('node', shape='circle', fontsize='11pt')

    for node in edge_index.view(-1).unique().tolist():
        g.node(str(node))

    for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        hex_color = hex(255 - round(255 * w))[2:]
        hex_color = f'{hex_color}0' if len(hex_color) == 1 else hex_color
        g.edge(str(src), str(dst), color=f'#{hex_color}{hex_color}{hex_color}')

    if path is not None:
        path = '.'.join(path.split('.')[:-1])
        g.render(path, cleanup=True)
    else:
        g.view()

    return g


def _visualize_graph_via_networkx(
    edge_index: Tensor,
    edge_weight: Tensor,
    path: Optional[str] = None,
) -> Any:
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()
    node_size = 10

    for node in edge_index.view(-1).unique().tolist():
        g.add_node(node)

    for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        g.add_edge(src, dst, alpha=w)

    ax = plt.gca()
    pos = nx.spring_layout(g)
    for src, dst, data in g.edges(data=True):
        ax.annotate(
            '',
            xy=pos[src],
            xytext=pos[dst],
            arrowprops=dict(
                arrowstyle="->",
                alpha=data['alpha'],
                shrinkA=sqrt(node_size) / 2.0,
                shrinkB=sqrt(node_size) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    nodes = nx.draw_networkx_nodes(g, pos, node_size=node_size,
                                   node_color='white', margins=0.1)
    nodes.set_edgecolor('black')
    nx.draw_networkx_labels(g, pos, font_size=10)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()