import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_npz(path):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        return data['y_true'], data['y_pred']
    except Exception:
        return None

def compute_bins(y_true, num_bins=6):
    qs = np.linspace(0, 1, num_bins + 1)
    edges = np.quantile(y_true, qs)
    edges[0] = np.minimum(edges[0], 0.0)
    return edges

def bin_metrics(y_true, y_pred, edges):
    bins = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_true >= lo) & (y_true <= hi)
        if not np.any(mask):
            bins.append({'range': (float(lo), float(hi)), 'rmse': None, 'mae': None, 'mape': None, 'count': 0})
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        mae = float(np.mean(np.abs(yp - yt)))
        yt_clip = np.clip(yt, 1e-6, None)
        mape = float(np.mean(np.abs((yp - yt) / yt_clip)))
        bins.append({'range': (float(lo), float(hi)), 'rmse': rmse, 'mae': mae, 'mape': mape, 'count': int(mask.sum())})
    return bins

def plot_bins(bins_dict, out_path):
    labels = []
    models = list(bins_dict.keys())
    for r in bins_dict[models[0]]:
        lo, hi = r['range']
        labels.append(f"[{lo:.0f},{hi:.0f}]")
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 4))
    for i, m in enumerate(models):
        maes = [b['mae'] if b['mae'] is not None else np.nan for b in bins_dict[m]]
        plt.plot(x, maes, marker='o', label=m)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel('MAE')
    plt.title('Error by Distance Bins (Test subset)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main(data_dir: str):
    artifacts = os.path.join(data_dir, 'artifacts')
    ensure_dir(os.path.join(artifacts, 'figures'))
    mlp_npz = os.path.join(artifacts, 'mlp_pred_coords_diff.npz')
    gnn_npz = os.path.join(artifacts, 'gnn_pred.npz')

    mlp = load_npz(mlp_npz)
    gnn = load_npz(gnn_npz)
    if mlp is None and gnn is None:
        print('[bins] no prediction files found')
        return
    y_true_ref = None
    if mlp is not None:
        y_true_ref = mlp[0]
    elif gnn is not None:
        y_true_ref = gnn[0]
    edges = compute_bins(y_true_ref, num_bins=6)
    out = {}
    if mlp is not None:
        out['mlp_coords_diff'] = bin_metrics(mlp[0], mlp[1], edges)
    if gnn is not None:
        out['gnn'] = bin_metrics(gnn[0], gnn[1], edges)
    fig_path = os.path.join(artifacts, 'figures', 'error_bins.png')
    plot_bins(out, fig_path)
    with open(os.path.join(artifacts, 'error_bins.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[bins] saved: {fig_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    args = ap.parse_args()
    main(args.data_dir)