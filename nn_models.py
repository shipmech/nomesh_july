import torch
import torch.nn as nn
from torch_geometric.nn import SplineConv

class TransformConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim, kernel_size, degree=1, aggr='add'):
        super().__init__()
        self.conv = SplineConv(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            kernel_size=kernel_size,
            degree=degree,
            root_weight=False,
            bias=True,
            aggr=aggr
        )

    def forward(self, x, edge_index, edge_attr, size):
        return self.conv(x, edge_index, edge_attr, size)

class BCTransformingMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, bc_data: torch.Tensor) -> torch.Tensor:
        return self.mlp(bc_data)