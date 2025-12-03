import os
import json
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from src.models.gnn_model import GraphDistanceModel
from src.data.utils import build_node_index, graph_to_edge_index, normalize_features


def load_graph_and_features(data_dir: str):
    import pickle
    with open(os.path.join(data_dir, 'graph.gpickle'), 'rb') as f:
        G = pickle.load(f)
    node_df = pd.read_csv(os.path.join(data_dir, 'node_features.csv'))
    lm_npz = np.load(os.path.join(data_dir, 'landmark_distances.npz'))
    lm_dists = lm_npz['distances']  # [N, K]
    return G, node_df, lm_dists


def build_feature_matrix(node_df: pd.DataFrame, lm_dists: np.ndarray) -> np.ndarray:
    # Basic features
    X_basic = node_df[['lat', 'lon', 'degree', 'pagerank', 'is_intersection']].values.astype(np.float32)
    # Concatenate landmark distances
    lm = lm_dists.astype(np.float32)
    # Replace non-finite landmark distances with NaN; downstream normalization handles NaN
    lm[~np.isfinite(lm)] = np.nan
    X = np.concatenate([X_basic, lm], axis=1)
    Xn, mean, std = normalize_features(X)
    return Xn


def split_pairs(df_pairs: pd.DataFrame, train_ratio=0.7, val_ratio=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(df_pairs))
    np.random.shuffle(idx)
    n_train = int(train_ratio * len(idx))
    n_val = int(val_ratio * len(idx))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return df_pairs.iloc[train_idx], df_pairs.iloc[val_idx], df_pairs.iloc[test_idx]


def evaluate(model, x, edge_index, pairs, y_true, inverse: str = 'none') -> Tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        y_pred = model(x, edge_index, pairs)
        y = y_true
        if inverse == 'log1p':
            y_pred = torch.expm1(y_pred)
            y = torch.expm1(y)
        rmse = torch.sqrt(torch.mean((y_pred - y) ** 2)).item()
        mae = torch.mean(torch.abs(y_pred - y)).item()
        mape = torch.mean(torch.abs((y_pred - y) / torch.clamp(y, min=1e-6))).item()
    return rmse, mae, mape


def main(data_dir: str, epochs: int = 50, hidden_dim: int = 64, num_layers: int = 3, lr: float = 1e-3, device: str = 'cpu', loss: str = 'mse', target_transform: str = 'none', weight_decay: float = 0.0, patience: int = 0):
    os.makedirs(os.path.join(data_dir, 'artifacts'), exist_ok=True)

    print("[GNN] Loading data...")
    G, node_df, lm_dists = load_graph_and_features(data_dir)
    df_pairs = pd.read_csv(os.path.join(data_dir, 'pairs.csv'))
    df_pairs = df_pairs.replace([np.inf, -np.inf], np.nan).dropna(subset=['d'])

    # Build node index mapping and edge_index
    node_to_idx = build_node_index(G)
    edge_index = graph_to_edge_index(G, node_to_idx)
    edge_index = torch.from_numpy(edge_index).long().to(device)

    # Build feature matrix in node order
    node_order = node_df['node'].tolist()
    # Reorder lm_dists according to node_df order already done by preprocess
    X = build_feature_matrix(node_df, lm_dists)
    x = torch.from_numpy(X).float().to(device)

    # Map pairs to index
    def to_pair_idx(df):
        u_idx = [node_to_idx[u] for u in df['u'].tolist()]
        v_idx = [node_to_idx[v] for v in df['v'].tolist()]
        return torch.tensor(np.stack([u_idx, v_idx], axis=1), dtype=torch.long).to(device)

    train_df, val_df, test_df = split_pairs(df_pairs)
    train_pairs = to_pair_idx(train_df)
    val_pairs = to_pair_idx(val_df)
    test_pairs = to_pair_idx(test_df)

    y_train = torch.tensor(train_df['d'].values, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_df['d'].values, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_df['d'].values, dtype=torch.float32).to(device)
    if target_transform == 'log1p':
        y_train = torch.log1p(y_train)
        y_val = torch.log1p(y_val)
        y_test = torch.log1p(y_test)

    in_dim = x.size(1)
    model = GraphDistanceModel(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss() if loss == 'mse' else nn.HuberLoss()
    loss_type = loss

    print("[GNN] Training...")
    t0 = time.time()
    best_val = float('inf')
    best_state = None
    wait = 0
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        y_pred = model(x, edge_index, train_pairs)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        opt.step()

        if ep % 5 == 0 or ep == 1:
            rmse, mae, mape = evaluate(model, x, edge_index, val_pairs, y_val, inverse=target_transform if target_transform == 'log1p' else 'none')
            print(f"Epoch {ep:03d} | TrainLoss {loss.item():.4f} | Val RMSE {rmse:.4f} MAE {mae:.4f} MAPE {mape:.4f}")
            if rmse < best_val:
                best_val = rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if patience > 0 and wait >= patience:
                    print("[GNN] Early stopping")
                    break

    train_time = time.time() - t0

    print("[GNN] Evaluating on test set...")
    if best_state is not None:
        model.load_state_dict(best_state)
    rmse, mae, mape = evaluate(model, x, edge_index, test_pairs, y_test, inverse=target_transform if target_transform == 'log1p' else 'none')
    print(f"Test RMSE {rmse:.4f} MAE {mae:.4f} MAPE {mape:.4f}")

    torch.save(model.state_dict(), os.path.join(data_dir, 'artifacts', 'gnn_model.pt'))
    with open(os.path.join(data_dir, 'artifacts', 'gnn_metrics.json'), 'w') as f:
        json.dump({'rmse': rmse, 'mae': mae, 'mape': mape, 'train_time_sec': train_time, 'loss': loss_type, 'target_transform': target_transform, 'weight_decay': weight_decay, 'patience': patience, 'hidden_dim': hidden_dim, 'num_layers': num_layers, 'epochs': ep}, f, indent=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'huber'])
    parser.add_argument('--target_transform', type=str, default='none', choices=['none', 'log1p'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=0)
    args = parser.parse_args()
    main(data_dir=args.data_dir, epochs=args.epochs, hidden_dim=args.hidden_dim, num_layers=args.num_layers, lr=args.lr, device=args.device, loss=args.loss, target_transform=args.target_transform, weight_decay=args.weight_decay, patience=args.patience)
