import torch
import torch.nn as nn
from torch_geometric.nn import SplineConv
from torch_geometric.nn import MessagePassing

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
    
import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class MessagePassingMLPConv(MessagePassing):
    def __init__(self, src_channels: int, out_channels: int, hidden_dim: int = 64, aggr: str = 'add'):
        super().__init__(aggr=aggr, flow='source_to_target')
        
        # MLP for message: processes source node features (base_j + bc_j)
        self.message_mlp = nn.Sequential(
            nn.Linear(src_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, x: tuple, edge_index: torch.Tensor, size: tuple = None) -> torch.Tensor:
        # x: (x_dst [num_dst, dst_channels], x_src [num_src, src_channels])
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        # x_j: [num_edges, src_channels]
        return self.message_mlp(x_j)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # aggr_out: [num_dst, out_channels], x: x_dst [num_dst, dst_channels]
        return aggr_out