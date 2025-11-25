from typing import List, Tuple, Dict
import random
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_node_features(G: nx.MultiDiGraph) -> pd.DataFrame:
    """Compute node features: lat, lon, degree, pagerank, is_intersection."""
    # Degree
    degrees = dict(G.degree())
    # PageRank (unweighted)
    pr = nx.pagerank(G, alpha=0.85)

    rows = []
    for n, d in G.nodes(data=True):
        lat = d.get('y', None)
        lon = d.get('x', None)
        deg = degrees.get(n, 0)
        is_intersection = 1 if deg > 2 else 0
        rows.append(
            {
                'node': n,
                'lat': lat,
                'lon': lon,
                'degree': deg,
                'pagerank': pr.get(n, 0.0),
                'is_intersection': is_intersection,
            }
        )

    df = pd.DataFrame(rows)
    return df


def select_landmarks(G: nx.MultiDiGraph, K: int) -> List[int]:
    """Select K landmark nodes by highest degree, as a simple heuristic."""
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    lm_nodes = [n for n, _deg in degrees[:K]]
    return lm_nodes


def compute_landmark_distances(G: nx.MultiDiGraph, landmarks: List[int]) -> Dict[int, List[float]]:
    """For each node, compute distances to each landmark using Dijkstra length."""
    node_to_dists = {n: [np.inf] * len(landmarks) for n in G.nodes()}
    for i, lm in enumerate(tqdm(landmarks, desc='Landmark distances')):
        dists = nx.single_source_dijkstra_path_length(G, source=lm, weight='length')
        for n, dist in dists.items():
            node_to_dists[n][i] = dist
    return node_to_dists


def sample_node_pairs(G: nx.MultiDiGraph, num_pairs: int) -> List[Tuple[int, int]]:
    """Uniformly sample node pairs (u, v), u != v."""
    nodes = list(G.nodes())
    pairs = []
    for _ in range(num_pairs):
        u, v = random.sample(nodes, 2)
        pairs.append((u, v))
    return pairs


def compute_ground_truth_distances(G: nx.MultiDiGraph, pairs: List[Tuple[int, int]]) -> List[float]:
    dists = []
    for (u, v) in tqdm(pairs, desc='Ground truth (Dijkstra)'):
        try:
            d = nx.shortest_path_length(G, source=u, target=v, weight='length')
        except nx.NetworkXNoPath:
            d = np.inf
        dists.append(d)
    return dists


def save_dataset(outdir: str,
                 G: nx.MultiDiGraph,
                 node_features: pd.DataFrame,
                 landmarks: List[int],
                 lm_dists: Dict[int, List[float]],
                 pairs: List[Tuple[int, int]],
                 gt_dists: List[float]) -> None:
    os.makedirs(outdir, exist_ok=True)
    # Graph
    import pickle
    with open(os.path.join(outdir, 'graph.gpickle'), 'wb') as f:
        pickle.dump(G, f)

    # Node features
    node_features.to_csv(os.path.join(outdir, 'node_features.csv'), index=False)

    # Landmark distances matrix aligned with node_features order
    node_order = node_features['node'].tolist()
    lm_mat = np.array([lm_dists[n] for n in node_order], dtype=np.float32)
    np.savez(os.path.join(outdir, 'landmark_distances.npz'),
             nodes=np.array(node_order, dtype=np.int64),
             distances=lm_mat,
             landmarks=np.array(landmarks, dtype=np.int64))

    # Pairs with ground truth
    df_pairs = pd.DataFrame({
        'u': [u for u, _ in pairs],
        'v': [v for _, v in pairs],
        'd': gt_dists,
    })
    df_pairs.to_csv(os.path.join(outdir, 'pairs.csv'), index=False)
