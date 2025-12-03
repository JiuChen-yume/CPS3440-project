import os
import argparse
import json
import math
import heapq
import numpy as np
import pandas as pd
import networkx as nx

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def edge_weight(G: nx.MultiDiGraph, u: int, v: int) -> float:
    ed = G.get_edge_data(u, v)
    if not ed:
        return math.inf
    w = math.inf
    for k, d in ed.items():
        w = min(w, d.get('length', math.inf))
    return w

def astar_with_counts(G: nx.MultiDiGraph, s: int, t: int):
    du = G.nodes[s]
    dt = G.nodes[t]
    lat_t, lon_t = dt.get('y'), dt.get('x')
    def h(n):
        dn = G.nodes[n]
        lat, lon = dn.get('y'), dn.get('x')
        if lat is None or lon is None or lat_t is None or lon_t is None:
            return 0.0
        return haversine_distance(lat, lon, lat_t, lon_t)
    openq = []
    heapq.heappush(openq, (h(s), 0.0, s))
    came = {}
    g = {s: 0.0}
    expanded = 0
    max_frontier = 1
    seen = set()
    while openq:
        f, gcur, u = heapq.heappop(openq)
        expanded += 1
        if u == t:
            return gcur, expanded, max_frontier
        if u in seen:
            continue
        seen.add(u)
        for v in G.successors(u):
            w = edge_weight(G, u, v)
            if not math.isfinite(w):
                continue
            ng = gcur + w
            if ng < g.get(v, math.inf):
                g[v] = ng
                came[v] = u
                heapq.heappush(openq, (ng + h(v), ng, v))
        max_frontier = max(max_frontier, len(openq))
    return math.inf, expanded, max_frontier

def dijkstra_with_counts(G: nx.MultiDiGraph, s: int, t: int):
    openq = []
    heapq.heappush(openq, (0.0, s))
    dist = {s: 0.0}
    expanded = 0
    max_frontier = 1
    seen = set()
    while openq:
        dcur, u = heapq.heappop(openq)
        expanded += 1
        if u == t:
            return dcur, expanded, max_frontier
        if u in seen:
            continue
        seen.add(u)
        for v in G.successors(u):
            w = edge_weight(G, u, v)
            if not math.isfinite(w):
                continue
            nd = dcur + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(openq, (nd, v))
        max_frontier = max(max_frontier, len(openq))
    return math.inf, expanded, max_frontier

def main(data_dir: str, sample_size: int = 200):
    import pickle
    with open(os.path.join(data_dir, 'graph.gpickle'), 'rb') as f:
        G = pickle.load(f)
    pairs_df = pd.read_csv(os.path.join(data_dir, 'pairs.csv'))
    pairs_df = pairs_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['d'])
    n = len(pairs_df)
    test_df = pairs_df.iloc[int(n * 0.85):]
    if len(test_df) > sample_size:
        test_df = test_df.sample(sample_size, random_state=42)
    stats = []
    for _, r in test_df.iterrows():
        s = int(r['u']); t = int(r['v'])
        d_dij, exp_dij, mem_dij = dijkstra_with_counts(G, s, t)
        d_ast, exp_ast, mem_ast = astar_with_counts(G, s, t)
        stats.append({
            'u': s, 'v': t,
            'dijkstra_dist': float(d_dij), 'a_star_dist': float(d_ast),
            'dijkstra_expanded': int(exp_dij), 'a_star_expanded': int(exp_ast),
            'dijkstra_max_frontier': int(mem_dij), 'a_star_max_frontier': int(mem_ast),
        })
    # Aggregate
    df = pd.DataFrame(stats)
    agg = {
        'pairs': len(df),
        'dijkstra_expanded_avg': float(df['dijkstra_expanded'].mean()),
        'a_star_expanded_avg': float(df['a_star_expanded'].mean()),
        'dijkstra_frontier_avg': float(df['dijkstra_max_frontier'].mean()),
        'a_star_frontier_avg': float(df['a_star_max_frontier'].mean()),
        'expansion_speedup': float(df['dijkstra_expanded'].mean() / max(df['a_star_expanded'].mean(), 1.0)),
    }
    artifacts = os.path.join(data_dir, 'artifacts')
    ensure_dir(artifacts)
    with open(os.path.join(artifacts, 'expansion_stats.json'), 'w') as f:
        json.dump({'aggregate': agg, 'samples': stats}, f, indent=2)
    # Simple bar figure
    import matplotlib.pyplot as plt
    labels = ['Dijkstra expanded', 'A* expanded', 'Dijkstra frontier', 'A* frontier']
    values = [agg['dijkstra_expanded_avg'], agg['a_star_expanded_avg'], agg['dijkstra_frontier_avg'], agg['a_star_frontier_avg']]
    plt.figure(figsize=(8,4))
    plt.bar(labels, values, color=['#4c72b0', '#dd8452', '#4c72b0', '#dd8452'])
    plt.ylabel('Average count')
    plt.title('Node Expansions and Frontier Size (avg, test sample)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig_path = os.path.join(artifacts, 'figures', 'expansion_bars.png')
    ensure_dir(os.path.dirname(fig_path))
    plt.savefig(fig_path)
    plt.close()
    print(f"[expansion] saved: {fig_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--sample_size', type=int, default=200)
    args = ap.parse_args()
    main(args.data_dir, sample_size=args.sample_size)