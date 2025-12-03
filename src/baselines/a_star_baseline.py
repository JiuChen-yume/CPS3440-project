from typing import List, Tuple
import time
import networkx as nx
import numpy as np
from src.data.utils import haversine_distance


def _heuristic(G: nx.MultiDiGraph, u: int, v: int) -> float:
    du = G.nodes[u]
    dv = G.nodes[v]
    lat1, lon1 = du.get('y'), du.get('x')
    lat2, lon2 = dv.get('y'), dv.get('x')
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0
    return haversine_distance(lat1, lon1, lat2, lon2)


def a_star_distances(G: nx.MultiDiGraph, pairs: List[Tuple[int, int]]) -> Tuple[np.ndarray, float]:
    """Compute shortest path distances via A* with geo heuristic per pair."""
    tic = time.time()
    out = []
    for u, v in pairs:
        try:
            d = nx.astar_path_length(G, u, v, heuristic=lambda a, b: _heuristic(G, a, b), weight='length')
        except nx.NetworkXNoPath:
            d = np.inf
        out.append(d)
    elapsed = time.time() - tic
    return np.array(out, dtype=np.float64), elapsed

