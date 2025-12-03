import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def plot_times(evaluation_summary_path: str, out_dir: str):
    if not os.path.exists(evaluation_summary_path):
        print(f"[plot] skip times: {evaluation_summary_path} not found")
        return
    with open(evaluation_summary_path, 'r') as f:
        summary = json.load(f)

    labels, times = [], []
    for k, v in summary.items():
        if 'inference_time_sec' in v:
            labels.append(k)
            times.append(v['inference_time_sec'])
    if not labels:
        print("[plot] no inference_time_sec in summary")
        return

    plt.figure(figsize=(8, 4))
    plt.bar(labels, times, color=['#4c72b0', '#dd8452', '#55a868', '#c44e52'])
    plt.ylabel('Inference Time (sec)')
    plt.title('Algorithm Inference Time (Test subset)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'inference_times.png'))
    plt.close()


def plot_mlp_metrics(artifacts_dir: str, out_dir: str):
    metric_files = [f for f in os.listdir(artifacts_dir) if f.startswith('mlp_metrics_') and f.endswith('.json')]
    if not metric_files:
        # fallback to single unsuffixed metrics
        fallback = os.path.join(artifacts_dir, 'mlp_metrics.json')
        if not os.path.exists(fallback):
            print("[plot] no mlp_metrics found")
            return
        with open(fallback, 'r') as f:
            m = json.load(f)
        labels, rmses, maes, mapes = ['coords_diff'], [m.get('rmse')], [m.get('mae')], [m.get('mape')]
        x = np.arange(len(labels))
        width = 0.25
        plt.figure(figsize=(6, 4))
        plt.bar(x - width, rmses, width, label='RMSE')
        plt.bar(x, maes, width, label='MAE')
        plt.bar(x + width, mapes, width, label='MAPE')
        plt.xticks(x, labels)
        plt.ylabel('Error')
        plt.title('MLP Metrics (single run)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mlp_metrics.png'))
        plt.close()
        return
    labels, rmses, maes, mapes = [], [], [], []
    for mf in metric_files:
        with open(os.path.join(artifacts_dir, mf), 'r') as f:
            m = json.load(f)
        label = mf.replace('mlp_metrics_', '').replace('.json', '')
        labels.append(label)
        rmses.append(m.get('rmse'))
        maes.append(m.get('mae'))
        mapes.append(m.get('mape'))

    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(9, 4))
    plt.bar(x - width, rmses, width, label='RMSE')
    plt.bar(x, maes, width, label='MAE')
    plt.bar(x + width, mapes, width, label='MAPE')
    plt.xticks(x, labels, rotation=15)
    plt.ylabel('Error')
    plt.title('MLP Metrics Across Feature Sets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mlp_metrics.png'))
    plt.close()


def plot_pred_scatter(artifacts_dir: str, out_dir: str, data_dir: str):
    pred_files = [f for f in os.listdir(artifacts_dir) if f.startswith('mlp_pred_') and f.endswith('.npz')]
    if not pred_files:
        # Try to generate predictions from saved model for default feature set
        model_path = os.path.join(artifacts_dir, 'mlp_model.pkl')
        if not os.path.exists(model_path):
            print("[plot] no mlp_pred files or model to generate predictions")
            return
        # Lazy import to avoid heavy deps at module import
        import joblib
        import pandas as pd
        # Recreate test split and inputs
        node_df = pd.read_csv(os.path.join(data_dir, 'node_features.csv'))
        df_pairs = pd.read_csv(os.path.join(data_dir, 'pairs.csv'))
        df_pairs = df_pairs.replace([np.inf, -np.inf], np.nan).dropna(subset=['d'])
        # Same split as training
        idx = np.arange(len(df_pairs))
        np.random.shuffle(idx)
        n_train = int(0.7 * len(idx))
        n_val = int(0.15 * len(idx))
        test_idx = idx[n_train + n_val:]
        test_df = df_pairs.iloc[test_idx]
        # Build inputs for default features (coords+diff)
        def build_inputs(node_df_local, df_pairs_local):
            node_to_coord = {row['node']: (row['lat'], row['lon']) for _, row in node_df_local.iterrows()}
            feats = []
            for _, r in df_pairs_local.iterrows():
                lu, lo = node_to_coord[r['u']]
                lv, l2o = node_to_coord[r['v']]
                dlat = lv - lu
                dlon = l2o - lo
                feats.append([lu, lo, lv, l2o, dlat, dlon])
            return np.array(feats, dtype=np.float32)
        X_test = build_inputs(node_df, test_df)
        y_test = test_df['d'].values
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        out_npz = os.path.join(artifacts_dir, 'mlp_pred_coords_diff.npz')
        np.savez_compressed(out_npz, y_true=y_test, y_pred=y_pred)
        pred_files = [os.path.basename(out_npz)]
    for pf in pred_files:
        data = np.load(os.path.join(artifacts_dir, pf))
        y_true = data['y_true']
        y_pred = data['y_pred']
        label = pf.replace('mlp_pred_', '').replace('.npz', '')

        # Scatter y_pred vs y_true
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true, y_pred, s=4, alpha=0.5)
        maxv = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
        plt.plot([0, maxv], [0, maxv], 'r--', linewidth=1)
        plt.xlabel('True distance (m)')
        plt.ylabel('Predicted distance (m)')
        plt.title(f'MLP Prediction Scatter ({label})')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'mlp_scatter_{label}.png'))
        plt.close()

        # Error vs true distance
        err = y_pred - y_true
        plt.figure(figsize=(6, 4))
        plt.scatter(y_true, err, s=3, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--', linewidth=1)
        plt.xlabel('True distance (m)')
        plt.ylabel('Error (m)')
        plt.title(f'MLP Error vs Distance ({label})')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'mlp_error_{label}.png'))
        plt.close()


def main(data_dir: str):
    artifacts = os.path.join(data_dir, 'artifacts')
    out_dir = os.path.join(artifacts, 'figures')
    ensure_dir(out_dir)

    eval_summary = os.path.join(data_dir, 'artifacts', 'evaluation_summary.json')
    plot_times(eval_summary, out_dir)
    plot_mlp_metrics(artifacts, out_dir)
    plot_pred_scatter(artifacts, out_dir, data_dir)
    print(f"[plot] figures saved to: {out_dir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    args = ap.parse_args()
    main(args.data_dir)
