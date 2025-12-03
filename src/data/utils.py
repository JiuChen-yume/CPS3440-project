from typing import Dict, Tuple
import math
import networkx as nx
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance in meters between two lat/lon points."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def build_node_index(G: nx.Graph) -> Dict[int, int]:
    """Map original node id to contiguous index [0..N-1]."""
    return {n: i for i, n in enumerate(G.nodes())}


def graph_to_edge_index(G: nx.Graph, node_to_idx: Dict[int, int]) -> np.ndarray:
    """Convert NetworkX graph edges to PyG-like edge_index (2 x E)."""
    edges = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            edges.append([node_to_idx[u], node_to_idx[v]])
    edge_index = np.array(edges, dtype=np.int64).T  # shape [2, E]
    return edge_index


def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust standardization with NaN/Inf handling.

    - Replaces Inf with NaN for aggregation.
    - Uses nanmean/nanstd to ignore NaNs when computing statistics.
    - Fills NaNs in X with per-feature mean before normalization.
    """
    X = X.copy()
    # Replace inf/-inf with NaN to exclude from stats
    X[~np.isfinite(X)] = np.nan
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    # Avoid zero std and NaN
    std = np.where((std == 0) | ~np.isfinite(std), 1.0, std)
    mean = np.where(~np.isfinite(mean), 0.0, mean)
    # Fill NaNs in X with mean
    inds = np.where(~np.isfinite(X))
    if inds[0].size > 0:
        X[inds] = np.take(mean, inds[1])
    Xn = (X - mean) / (std + 1e-8)
    return Xn, mean, std
