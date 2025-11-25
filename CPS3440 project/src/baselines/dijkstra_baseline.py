from typing import List, Tuple
import time
import networkx as nx
import numpy as np


def dijkstra_distances(G: nx.MultiDiGraph, pairs: List[Tuple[int, int]]) -> Tuple[np.ndarray, float]:
    """Compute exact shortest path distances via Dijkstra for each pair.

    Returns distances array (meters) and elapsed time (seconds).
    """
    tic = time.time()
    out = []
    for u, v in pairs:
        try:
            d = nx.shortest_path_length(G, source=u, target=v, weight='length')
        except nx.NetworkXNoPath:
            d = np.inf
        out.append(d)
    elapsed = time.time() - tic
    return np.array(out, dtype=np.float64), elapsed

