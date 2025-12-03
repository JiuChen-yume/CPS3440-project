import os
import time
import json
from typing import Dict

import numpy as np
import pandas as pd
import networkx as nx
from src.baselines.dijkstra_baseline import dijkstra_distances
from src.baselines.a_star_baseline import a_star_distances
from src.data.utils import build_node_index, graph_to_edge_index, normalize_features
from src.evaluation.metrics import rmse, mae, mape


def evaluate_all(data_dir: str, device: str = 'cpu') -> Dict:
    import pickle
    with open(os.path.join(data_dir, 'graph.gpickle'), 'rb') as f:
        G = pickle.load(f)
    node_df = pd.read_csv(os.path.join(data_dir, 'node_features.csv'))
    pairs_df = pd.read_csv(os.path.join(data_dir, 'pairs.csv'))
    # Remove unreachable pairs with infinite distances
    pairs_df = pairs_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['d'])
    # Split test subset (last 15%)
    n = len(pairs_df)
    test_df = pairs_df.iloc[int(n * 0.85):]
    pairs = [(int(u), int(v)) for u, v in zip(test_df['u'], test_df['v'])]
    y_true = test_df['d'].values.astype(np.float64)

    results = {}

    # Baselines
    dijkstra_pred, d_time = dijkstra_distances(G, pairs)
    results['dijkstra'] = {
        'rmse': rmse(y_true, dijkstra_pred),
        'mae': mae(y_true, dijkstra_pred),
        'mape': mape(y_true, dijkstra_pred),
        'inference_time_sec': d_time,
    }

    a_star_pred, a_time = a_star_distances(G, pairs)
    results['a_star'] = {
        'rmse': rmse(y_true, a_star_pred),
        'mae': mae(y_true, a_star_pred),
        'mape': mape(y_true, a_star_pred),
        'inference_time_sec': a_time,
    }

    # MLP inference speed (robust: try suffixed models, then fallback to metrics/preds)
    artifacts_dir = os.path.join(data_dir, 'artifacts')
    def _eval_mlp_from_model(model_path: str):
        try:
            import joblib
            model = joblib.load(model_path)
            node_to_coord = {row['node']: (row['lat'], row['lon']) for _, row in node_df.iterrows()}
            X = []
            for (u, v) in pairs:
                lu, lo = node_to_coord[u]
                lv, l2o = node_to_coord[v]
                dlat = lv - lu
                dlon = l2o - lo
                X.append([lu, lo, lv, l2o, dlat, dlon])
            X = np.array(X, dtype=np.float32)
            tic = time.time()
            yp = model.predict(X)
            inf_t = time.time() - tic
            return {
                'rmse': rmse(y_true, yp),
                'mae': mae(y_true, yp),
                'mape': mape(y_true, yp),
                'inference_time_sec': inf_t,
            }
        except Exception as e:
            print(f"[WARN] Failed to load MLP model at {model_path}: {e}")
            return None

    mlp_candidates = [
        os.path.join(artifacts_dir, 'mlp_model_coords_diff.pkl'),
        os.path.join(artifacts_dir, 'mlp_model_coords.pkl'),
        os.path.join(artifacts_dir, 'mlp_model.pkl'),
    ]
    mlp_result = None
    for mp in mlp_candidates:
        if os.path.exists(mp):
            mlp_result = _eval_mlp_from_model(mp)
            if mlp_result is not None:
                break
    if mlp_result is None:
        # Fallback to precomputed metrics or predictions if available
        metrics_path = os.path.join(artifacts_dir, 'mlp_metrics.json')
        preds_path = os.path.join(artifacts_dir, 'mlp_pred_coords_diff.npz')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    m = json.load(f)
                mlp_result = {
                    'rmse': float(m.get('rmse')),
                    'mae': float(m.get('mae')),
                    'mape': float(m.get('mape')),
                }
            except Exception:
                mlp_result = None
        if mlp_result is None and os.path.exists(preds_path):
            try:
                data = np.load(preds_path)
                yp = data['y_pred']
                yt = data['y_true']
                mlp_result = {
                    'rmse': rmse(yt, yp),
                    'mae': mae(yt, yp),
                    'mape': mape(yt, yp),
                }
            except Exception:
                mlp_result = None
    if mlp_result is not None:
        results['mlp'] = mlp_result

    # GNN inference speed (if model exists)
    gnn_path = os.path.join(data_dir, 'artifacts', 'gnn_model.pt')
    lm_npz_path = os.path.join(data_dir, 'landmark_distances.npz')
    if os.path.exists(gnn_path) and os.path.exists(lm_npz_path):
        try:
            import torch
            from src.models.gnn_model import GraphDistanceModel
        except Exception:
            torch = None
        if torch is not None:
            lm_npz = np.load(lm_npz_path)
            lm_dists = lm_npz['distances']
            X_basic = node_df[['lat', 'lon', 'degree', 'pagerank', 'is_intersection']].values.astype(np.float32)
            lm = lm_dists.astype(np.float32)
            lm[~np.isfinite(lm)] = np.nan
            X = np.concatenate([X_basic, lm], axis=1)
            Xn, _, _ = normalize_features(X)
            x = torch.tensor(Xn, dtype=torch.float32).to(device)
            node_to_idx = build_node_index(G)
            edge_index = torch.tensor(graph_to_edge_index(G, node_to_idx), dtype=torch.long).to(device)
            pair_idx = torch.tensor([[node_to_idx[u], node_to_idx[v]] for (u, v) in pairs], dtype=torch.long).to(device)

            # Infer model hyperparameters from checkpoint
            sd = torch.load(gnn_path, map_location=device)
            try:
                hidden_dim = sd['node_encoder.0.weight'].shape[0]
            except Exception:
                hidden_dim = 64
            try:
                conv_indices = set()
                for k in sd.keys():
                    if k.startswith('convs.'):
                        idx = int(k.split('.')[1])
                        conv_indices.add(idx)
                num_layers = (max(conv_indices) + 1) if conv_indices else 3
            except Exception:
                num_layers = 3

            in_dim = x.size(1)
            model = GraphDistanceModel(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
            model.load_state_dict(sd)
            model.eval()
            inv = 'none'
            gm_path = os.path.join(data_dir, 'artifacts', 'gnn_metrics.json')
            if os.path.exists(gm_path):
                try:
                    gm = json.load(open(gm_path))
                    if gm.get('target_transform') == 'log1p':
                        inv = 'log1p'
                except Exception:
                    inv = 'none'
            tic = time.time()
            with torch.no_grad():
                yp = model(x, edge_index, pair_idx)
                if inv == 'log1p':
                    yp = torch.expm1(yp)
                yp = yp.cpu().numpy()
            inf_t = time.time() - tic
            results['gnn'] = {
                'rmse': rmse(y_true, yp),
                'mae': mae(y_true, yp),
                'mape': mape(y_true, yp),
                'inference_time_sec': inf_t,
            }
            try:
                np.savez_compressed(os.path.join(data_dir, 'artifacts', 'gnn_pred.npz'), y_true=y_true, y_pred=yp)
            except Exception:
                pass

    # Save
    os.makedirs(os.path.join(data_dir, 'artifacts'), exist_ok=True)
    with open(os.path.join(data_dir, 'artifacts', 'evaluation_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    out = evaluate_all(args.data_dir, device=args.device)
    print(json.dumps(out, indent=2))
