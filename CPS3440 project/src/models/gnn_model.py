from typing import Optional
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SAGEConv
except Exception as e:
    SAGEConv = None


class GraphDistanceModel(nn.Module):
    """GraphSAGE encoder + MLP predictor for pair distance."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for GraphDistanceModel. Please install torch_geometric.")
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pair_index: torch.Tensor) -> torch.Tensor:
        h = self.node_encoder(x)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout(h)
        hu = h[pair_index[:, 0]]
        hv = h[pair_index[:, 1]]
        z = torch.cat([hu, hv], dim=1)
        out = self.predictor(z).squeeze(-1)
        return out

