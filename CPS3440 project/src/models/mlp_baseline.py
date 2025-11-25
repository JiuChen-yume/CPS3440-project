import torch
import torch.nn as nn


class PairMLPRegressor(nn.Module):
    """Simple MLP to predict distance from raw coordinates of (u, v).

    Input features: [lat_u, lon_u, lat_v, lon_v, dlat, dlon]
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

