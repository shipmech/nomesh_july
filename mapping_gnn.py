import torch
import torch.nn as nn
from torch_geometric.nn import SplineConv, knn_graph
from torch_geometric.data import Data
from torch_geometric.nn.pool import knn
from torch_spline_conv import spline_basis
from config import SimulationConfig

class MappingGNN(nn.Module):
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.dim = 2
        self.register_buffer('kernel_size', torch.tensor([config.spline_kernel_size] * self.dim, dtype=torch.long))
        self.register_buffer('is_open_spline', torch.tensor([True] * self.dim, dtype=torch.uint8))
        # SplineConv for quantities (u, v, p, u_t, v_t, p_t)
        self.spline_conv_quantities = SplineConv(
            in_channels=6,
            out_channels=6,
            dim=self.dim,
            kernel_size=config.spline_kernel_size,
            degree=config.spline_degree,
            aggr=config.spline_aggr,
            root_weight=False
        )
        # SplineConv for first derivatives (separate for x and y directions)
        self.spline_conv_x = nn.ModuleList([
            SplineConv(
                in_channels=1,
                out_channels=1,
                dim=self.dim,
                kernel_size=config.spline_kernel_size,
                degree=config.spline_degree,
                aggr=config.spline_aggr,
                root_weight=False
            ) for _ in range(3)  # For u, v, p (x direction)
        ])
        self.spline_conv_y = nn.ModuleList([
            SplineConv(
                in_channels=1,
                out_channels=1,
                dim=self.dim,
                kernel_size=config.spline_kernel_size,
                degree=config.spline_degree,
                aggr=config.spline_aggr,
                root_weight=False
            ) for _ in range(3)  # For u, v, p (y direction)
        ])
        # SplineConv for second derivatives (one per scalar field: u, v)
        self.spline_conv_second = nn.ModuleList([
            SplineConv(
                in_channels=1,
                out_channels=1,
                dim=self.dim,
                kernel_size=config.spline_kernel_size,
                degree=config.spline_degree,
                aggr=config.spline_aggr,
                root_weight=False
            ) for _ in range(2)  # For u, v
        ])

    def forward(self, graph_data: Data, target_points: torch.Tensor, ensure_all_nodes: bool = False) -> Data:
        x = graph_data.x[:, -6:]  # Extract quantities (u, v, p, u_t, v_t, p_t)
        pos = graph_data.pos
        num_nodes = graph_data.num_nodes

        # Validate input sizes
        if x.size(0) != num_nodes:
            raise ValueError(f"Mismatch in number of nodes: x.size(0)={x.size(0)} vs graph_data.num_nodes={num_nodes}")
        if pos.size(0) != num_nodes:
            raise ValueError(f"Mismatch in number of nodes: pos.size(0)={pos.size(0)} vs graph_data.num_nodes={num_nodes}")

        # Compute graph edges for convolution
        edge_index = knn_graph(pos, k=self.config.k_neighbors, flow="source_to_target")
        edge_attr = (pos[edge_index[1]] - pos[edge_index[0]]) / self.config.radius
        edge_attr = torch.clamp(edge_attr, 0, 1)
        # Ensure edge_index is valid
        if edge_index.size(1) > 0 and (edge_index.max() >= num_nodes or edge_index.min() < 0):
            raise ValueError(f"Invalid edge_index: max={edge_index.max().item()} vs num_nodes={num_nodes}, min={edge_index.min().item()}")

        # Compute quantities using SplineConv
        conv_out = self.spline_conv_quantities(x, edge_index, edge_attr)

        # Interpolate to target points
        target_edge_index = knn(x=pos, y=target_points, k=self.config.k_neighbors)
        target_edge_attr = (pos[target_edge_index[1]] - target_points[target_edge_index[0]]) / self.config.radius
        target_edge_attr = torch.clamp(target_edge_attr, 0, 1)
        basis, weight = spline_basis(target_edge_attr, self.kernel_size, self.is_open_spline, self.config.spline_degree)
        # Compute interpolation weights
        interp_weights = (basis * weight).sum(dim=1, keepdim=True)  # Shape: [num_edges, 1]
        interp_weights = interp_weights / interp_weights.sum(dim=0, keepdim=True).clamp(min=1e-6)

        # Interpolate features to target points
        interpolated_x = torch.zeros(target_points.size(0), conv_out.size(1), device=conv_out.device)
        interpolated_x.scatter_add_(0, target_edge_index[0].unsqueeze(-1).expand(-1, conv_out.size(1)),
                                   conv_out[target_edge_index[1]] * interp_weights)

        # Adjust for ensure_all_nodes (optional, typically for graph_data alignment)
        full_pos = target_points  # Default to target_points for VTK grid (2500 points)
        if ensure_all_nodes and num_nodes != target_points.size(0):
            if num_nodes < target_points.size(0):
                interpolated_x = interpolated_x[:num_nodes]
                full_pos = target_points[:num_nodes]
            else:
                # Pad to match graph_data.num_nodes (for consistency with graph_data)
                padding_size = num_nodes - target_points.size(0)
                padded_x = torch.cat([interpolated_x, torch.zeros(padding_size, conv_out.size(1), device=conv_out.device)], dim=0)
                full_pos = torch.cat([pos, target_points[target_points.size(0):num_nodes]], dim=0) if padding_size > 0 else pos
                # Recompute for full set
                full_edge_index = knn_graph(full_pos, k=self.config.k_neighbors, flow="source_to_target")
                full_edge_attr = (full_pos[full_edge_index[1]] - full_pos[full_edge_index[0]]) / self.config.radius
                full_edge_attr = torch.clamp(full_edge_attr, 0, 1)
                conv_out_full = self.spline_conv_quantities(x, full_edge_index, full_edge_attr)
                full_target_edge_index = knn(x=full_pos, y=full_pos, k=self.config.k_neighbors)
                full_target_edge_attr = (full_pos[full_target_edge_index[1]] - full_pos[full_target_edge_index[0]]) / self.config.radius
                full_target_edge_attr = torch.clamp(full_target_edge_attr, 0, 1)
                full_basis, full_weight = spline_basis(full_target_edge_attr, self.kernel_size, self.is_open_spline, self.config.spline_degree)
                full_interp_weights = (full_basis * full_weight).sum(dim=1, keepdim=True)
                full_interp_weights = full_interp_weights / full_interp_weights.sum(dim=0, keepdim=True).clamp(min=1e-6)
                interpolated_x = torch.zeros(num_nodes, conv_out.size(1), device=conv_out.device)
                interpolated_x.scatter_add_(0, full_target_edge_index[0].unsqueeze(-1).expand(-1, conv_out.size(1)),
                                          conv_out_full[full_target_edge_index[1]] * full_interp_weights)

        print(f"MappingGNN: interpolated_x.size={interpolated_x.size()}, full_pos.size={full_pos.size()}, num_nodes={num_nodes}, target_points.size={target_points.size()}")

        # Compute derivative edges based on the final full_pos
        target_edge_index_deriv = knn_graph(full_pos, k=self.config.k_neighbors, flow="source_to_target")
        target_edge_attr_deriv = (full_pos[target_edge_index_deriv[1]] - full_pos[target_edge_index_deriv[0]]) / self.config.radius
        target_edge_attr_deriv = torch.clamp(target_edge_attr_deriv, 0, 1)
        basis_deriv, weight_deriv = spline_basis(target_edge_attr_deriv, self.kernel_size, self.is_open_spline, self.config.spline_degree)

        # Compute spatial derivatives
        u, v, p = interpolated_x[:, 0:1], interpolated_x[:, 1:2], interpolated_x[:, 2:3]
        u_x = self.spline_conv_x[0](u, target_edge_index_deriv, target_edge_attr_deriv)
        u_y = self.spline_conv_y[0](u, target_edge_index_deriv, target_edge_attr_deriv)
        v_x = self.spline_conv_x[1](v, target_edge_index_deriv, target_edge_attr_deriv)
        v_y = self.spline_conv_y[1](v, target_edge_index_deriv, target_edge_attr_deriv)
        p_x = self.spline_conv_x[2](p, target_edge_index_deriv, target_edge_attr_deriv)
        p_y = self.spline_conv_y[2](p, target_edge_index_deriv, target_edge_attr_deriv)

        # Second derivatives
        u_xx = self.spline_conv_second[0](u_x, target_edge_index_deriv, target_edge_attr_deriv)
        u_yy = self.spline_conv_second[0](u_y, target_edge_index_deriv, target_edge_attr_deriv)
        v_xx = self.spline_conv_second[1](v_x, target_edge_index_deriv, target_edge_attr_deriv)
        v_yy = self.spline_conv_second[1](v_y, target_edge_index_deriv, target_edge_attr_deriv)

        # Combine quantities and derivatives
        derivatives = torch.cat([u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy], dim=-1)
        return Data(pos=full_pos, x=interpolated_x, derivatives=derivatives)