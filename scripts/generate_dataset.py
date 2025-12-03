import argparse
import os
from src.data.download_graph import download_graph, largest_connected_component, ensure_node_xy, save_graph
from src.data.preprocess import (
    compute_node_features,
    select_landmarks,
    compute_landmark_distances,
    sample_node_pairs,
    compute_ground_truth_distances,
    save_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Download OSM graph and generate training dataset.")
    parser.add_argument('--place', type=str, required=True, help='Place name for OSMnx (e.g., "San Francisco, California, USA")')
    parser.add_argument('--network_type', type=str, default='drive', choices=['drive', 'walk', 'bike'])
    parser.add_argument('--num_pairs', type=int, default=20000, help='Number of node pairs to sample')
    parser.add_argument('--landmarks', type=int, default=16, help='Number of landmark nodes')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory to save dataset')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[1/5] Downloading graph for: {args.place}")
    G = download_graph(args.place, args.network_type)
    G = largest_connected_component(G, strongly=False)
    G = ensure_node_xy(G)

    print(f"[2/5] Computing node features")
    node_features = compute_node_features(G)

    print(f"[3/5] Selecting {args.landmarks} landmarks and computing distances")
    lm_nodes = select_landmarks(G, args.landmarks)
    lm_dists = compute_landmark_distances(G, lm_nodes)

    print(f"[4/5] Sampling {args.num_pairs} node pairs and computing ground truth distances")
    pairs = sample_node_pairs(G, args.num_pairs)
    gt_dists = compute_ground_truth_distances(G, pairs)

    print(f"[5/5] Saving dataset to {args.outdir}")
    save_dataset(args.outdir, G, node_features, lm_nodes, lm_dists, pairs, gt_dists)
    print("Done.")


if __name__ == '__main__':
    main()

