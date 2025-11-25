from typing import Optional

import osmnx as ox
import networkx as nx


def download_graph(place: str, network_type: str = "drive", simplify: bool = True) -> nx.MultiDiGraph:
    """
    Download a road network graph from OpenStreetMap using OSMnx.

    Args:
        place: A place name understood by OSMnx, e.g. "San Francisco, California, USA".
        network_type: One of {"drive", "walk", "bike"}.
        simplify: Whether to simplify the graph via OSMnx's built-in logic.

    Returns:
        A NetworkX MultiDiGraph with edge lengths added.
    """
    G = ox.graph_from_place(place, network_type=network_type, simplify=simplify)
    # OSMnx >=2.0 usually includes 'length' attribute on edges by default.
    # We keep as-is; downstream code will use 'length'.
    return G


def largest_connected_component(G: nx.MultiDiGraph, strongly: bool = False) -> nx.MultiDiGraph:
    """Keep the largest (weakly/strongly) connected component of the graph (NetworkX-based)."""
    if strongly:
        comps = list(nx.strongly_connected_components(G))
    else:
        comps = list(nx.connected_components(G.to_undirected()))
    if not comps:
        return G
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()


def ensure_node_xy(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Ensure every node has 'x' (lon) and 'y' (lat) attributes."""
    # OSMnx should populate these, but double-check
    for n, d in G.nodes(data=True):
        if 'x' not in d or 'y' not in d:
            # Fallback: remove nodes without coordinates
            G.remove_node(n)
    return G


def ensure_edge_lengths(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Ensure every edge has a 'length' attribute; if missing, approximate by haversine between nodes."""
    missing = 0
    for u, v, data in G.edges(data=True):
        if 'length' not in data:
            du = G.nodes[u]
            dv = G.nodes[v]
            lat1, lon1 = du.get('y'), du.get('x')
            lat2, lon2 = dv.get('y'), dv.get('x')
            if lat1 is not None and lon1 is not None and lat2 is not None and lon2 is not None:
                # simple haversine approximation
                from math import radians, sin, cos, atan2, sqrt
                R = 6371000.0
                phi1 = radians(lat1); phi2 = radians(lat2)
                dphi = radians(lat2 - lat1); dlambda = radians(lon2 - lon1)
                a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
                data['length'] = 2 * R * atan2(sqrt(a), sqrt(1-a))
                missing += 1
    return G


def save_graph(G: nx.MultiDiGraph, path: str) -> None:
    """Save graph as a gpickle for later reuse."""
    nx.write_gpickle(G, path)
