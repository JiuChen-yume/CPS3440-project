import os
import json
import time
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


def split_pairs(df_pairs: pd.DataFrame, train_ratio=0.7, val_ratio=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(df_pairs))
    np.random.shuffle(idx)
    n_train = int(train_ratio * len(idx))
    n_val = int(val_ratio * len(idx))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return df_pairs.iloc[train_idx], df_pairs.iloc[val_idx], df_pairs.iloc[test_idx]


def build_inputs(node_df: pd.DataFrame, df_pairs: pd.DataFrame, features: str = 'coords+diff') -> np.ndarray:
    node_to_coord = {row['node']: (row['lat'], row['lon']) for _, row in node_df.iterrows()}
    feats = []
    for _, r in df_pairs.iterrows():
        lu, lo = node_to_coord[r['u']]
        lv, l2o = node_to_coord[r['v']]
        if features == 'coords':
            feats.append([lu, lo, lv, l2o])
        else:
            dlat = lv - lu
            dlon = l2o - lo
            feats.append([lu, lo, lv, l2o, dlat, dlon])
    return np.array(feats, dtype=np.float32)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.clip(y_true, 1e-6, None)
    return float(np.mean(np.abs((y_pred - y_true) / y_true)))


def main(data_dir: str, epochs: int = 50, hidden_dim: int = 64, lr: float = 1e-3, device: str = 'cpu', features: str = 'coords+diff'):
    os.makedirs(os.path.join(data_dir, 'artifacts'), exist_ok=True)

    node_df = pd.read_csv(os.path.join(data_dir, 'node_features.csv'))
    df_pairs = pd.read_csv(os.path.join(data_dir, 'pairs.csv'))
    # Filter out unreachable pairs (inf distances)
    df_pairs = df_pairs.replace([np.inf, -np.inf], np.nan).dropna(subset=['d'])
    train_df, val_df, test_df = split_pairs(df_pairs)

    X_train = build_inputs(node_df, train_df, features=features)
    X_val = build_inputs(node_df, val_df, features=features)
    X_test = build_inputs(node_df, test_df, features=features)

    y_train = train_df['d'].values
    y_val = val_df['d'].values
    y_test = test_df['d'].values

    # sklearn pipeline with standardization + MLP
    hidden = hidden_dim
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(hidden, hidden), activation='relu', solver='adam', max_iter=epochs, learning_rate_init=lr, random_state=42))
    ])

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_val_pred = model.predict(X_val)
    rmse_val = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    mae_val = float(mean_absolute_error(y_val, y_val_pred))
    mape_val = mape(y_val, y_val_pred)
    print(f"[MLP] Val RMSE {rmse_val:.4f} MAE {mae_val:.4f} MAPE {mape_val:.4f}")

    y_test_pred = model.predict(X_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    mae_test = float(mean_absolute_error(y_test, y_test_pred))
    mape_test = mape(y_test, y_test_pred)
    print(f"[MLP] Test RMSE {rmse_test:.4f} MAE {mae_test:.4f} MAPE {mape_test:.4f}")

    # Save model, metrics, and predictions with feature suffix for comparison
    suffix = 'coords' if features == 'coords' else 'coords_diff'
    joblib.dump(model, os.path.join(data_dir, 'artifacts', f'mlp_model_{suffix}.pkl'))
    with open(os.path.join(data_dir, 'artifacts', f'mlp_metrics_{suffix}.json'), 'w') as f:
        json.dump({'rmse': rmse_test, 'mae': mae_test, 'mape': mape_test, 'train_time_sec': train_time,
                   'features': features, 'epochs': epochs, 'hidden_dim': hidden_dim, 'lr': lr}, f, indent=2)
    np.savez_compressed(os.path.join(data_dir, 'artifacts', f'mlp_pred_{suffix}.npz'),
                        y_true=y_test, y_pred=y_test_pred)


    if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type=str, required=True)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--device', type=str, default='cpu')
        parser.add_argument('--features', type=str, default='coords+diff', choices=['coords', 'coords+diff'])
        args = parser.parse_args()
        main(data_dir=args.data_dir, epochs=args.epochs, hidden_dim=args.hidden_dim, lr=args.lr, device=args.device, features=args.features)
