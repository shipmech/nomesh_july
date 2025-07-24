import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from config import SimulationConfig
from mesh import update_edges

class ICImprintingNN(nn.Module):
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(6, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.number_of_base_latent_features)
        )

    def forward(self, physical_quantities: torch.Tensor) -> torch.Tensor:
        return self.mlp(physical_quantities)

class BCTransformingNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, out_channels)
        )

    def forward(self, bc_data: torch.Tensor) -> torch.Tensor:
        return self.mlp(bc_data)

class BCCorrectionNN(nn.Module):
    def __init__(self, config: SimulationConfig, bc_features_dim: int):
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(config.number_of_base_latent_features * 2 + bc_features_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.number_of_base_latent_features)
        )

    def forward(self, base_features: torch.Tensor, aggregated_messages: torch.Tensor, bc_features: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([base_features, aggregated_messages, bc_features], dim=-1)
        return self.mlp(inputs)

class DynamicsGNN(MessagePassing):
    def __init__(self, config: SimulationConfig):
        super().__init__(aggr="add", node_dim=0)
        self.config = config
        self.node_types = ["free", "pressure", "velocity"]
        feature_dims = {
            "free": config.number_of_base_latent_features,
            "pressure": config.number_of_base_latent_features + config.number_of_pressure_bc_latent_features,
            "velocity": config.number_of_base_latent_features + config.number_of_velocities_bc_latent_features
        }

        self.message_nns = nn.ModuleDict({
            src: nn.ModuleDict({
                tgt: nn.Sequential(
                    nn.Linear(feature_dims[src] + feature_dims[tgt] + 1, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.number_of_base_latent_features)
                ) for tgt in self.node_types
            }) for src in self.node_types
        })

        self.to_physic_transformation_nns = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(config.number_of_base_latent_features * 2 + (config.number_of_pressure_bc_latent_features if nt == "pressure" else config.number_of_velocities_bc_latent_features if nt == "velocity" else 0), config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 6)
            ) for nt in self.node_types
        })

        self.bc_correction_pressure = BCCorrectionNN(config, config.number_of_pressure_bc_latent_features)
        self.bc_correction_velocity = BCCorrectionNN(config, config.number_of_velocities_bc_latent_features)
        self.bc_transform_pressure = BCTransformingNN(1, config.number_of_pressure_bc_latent_features, config)
        self.bc_transform_velocity = BCTransformingNN(2, config.number_of_velocities_bc_latent_features, config)

    def forward(self, graph_data: Data) -> Data:
        graph_data = update_edges(graph_data, self.config)
        node_type = graph_data.node_type.view(-1, 1)
        x = self.propagate(graph_data.edge_index, x=graph_data.x, pos=graph_data.pos, node_type=node_type, boundary_data=graph_data.boundary_data)
        graph_data.x = x
        return graph_data

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, pos_i: torch.Tensor, pos_j: torch.Tensor, node_type_i: torch.Tensor, node_type_j: torch.Tensor) -> torch.Tensor:
        messages = torch.zeros_like(x_i[:, :self.config.number_of_base_latent_features], dtype=x_i.dtype)
        feature_dims = {
            "free": self.config.number_of_base_latent_features,
            "pressure": self.config.number_of_base_latent_features + self.config.number_of_pressure_bc_latent_features,
            "velocity": self.config.number_of_base_latent_features + self.config.number_of_velocities_bc_latent_features
        }
        for i, src_type in enumerate(self.node_types):
            for j, tgt_type in enumerate(self.node_types):
                mask = (node_type_i.squeeze(-1) == i) & (node_type_j.squeeze(-1) == j)
                if mask.any():
                    src_features = x_i[mask, :feature_dims[src_type]]
                    tgt_features = x_j[mask, :feature_dims[tgt_type]]
                    dist = torch.norm(pos_j[mask] - pos_i[mask], dim=-1, keepdim=True)
                    inputs = torch.cat([src_features, tgt_features, dist], dim=-1)
                    messages[mask] = self.message_nns[src_type][tgt_type](inputs)
        return messages

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor, node_type: torch.Tensor, boundary_data: torch.Tensor) -> torch.Tensor:
        base_features = x[:, :self.config.number_of_base_latent_features]
        new_features = torch.zeros_like(x)
        new_features[:, :self.config.number_of_base_latent_features] = base_features

        pressure_mask = node_type.squeeze(-1) == 1
        velocity_mask = node_type.squeeze(-1) == 2
        new_features[pressure_mask, self.config.number_of_base_latent_features:self.config.number_of_base_latent_features + self.config.number_of_pressure_bc_latent_features] = self.bc_transform_pressure(boundary_data[pressure_mask, :1])
        new_features[velocity_mask, self.config.number_of_base_latent_features:self.config.number_of_base_latent_features + self.config.number_of_velocities_bc_latent_features] = self.bc_transform_velocity(boundary_data[velocity_mask, :2])

        derivatives = torch.zeros_like(base_features)
        derivatives[node_type.squeeze(-1) == 0] = aggr_out[node_type.squeeze(-1) == 0]
        derivatives[pressure_mask] = self.bc_correction_pressure(base_features[pressure_mask], aggr_out[pressure_mask], new_features[pressure_mask, self.config.number_of_base_latent_features:self.config.number_of_base_latent_features + self.config.number_of_pressure_bc_latent_features])
        derivatives[velocity_mask] = self.bc_correction_velocity(base_features[velocity_mask], aggr_out[velocity_mask], new_features[velocity_mask, self.config.number_of_base_latent_features:self.config.number_of_base_latent_features + self.config.number_of_velocities_bc_latent_features])

        phys_quantities = torch.zeros(x.shape[0], 6, device=x.device)
        for i, nt in enumerate(self.node_types):
            mask = node_type.squeeze(-1) == i
            if mask.any():
                bc_features = new_features[mask, self.config.number_of_base_latent_features:self.config.number_of_base_latent_features + (self.config.number_of_pressure_bc_latent_features if nt == "pressure" else self.config.number_of_velocities_bc_latent_features if nt == "velocity" else 0)]
                inputs = torch.cat([base_features[mask], derivatives[mask], bc_features], dim=-1)
                phys_quantities[mask] = self.to_physic_transformation_nns[nt](inputs)

        new_features[:, -6:] = phys_quantities
        return new_features

    def compute_derivatives(self, graph_data: Data) -> torch.Tensor:
        edge_index = graph_data.edge_index
        x = graph_data.x
        pos = graph_data.pos
        node_type = graph_data.node_type.view(-1, 1)
        boundary_data = graph_data.boundary_data
        new_features = torch.zeros_like(x)
        pressure_mask = node_type.squeeze(-1) == 1
        velocity_mask = node_type.squeeze(-1) == 2
        new_features[pressure_mask, self.config.number_of_base_latent_features:self.config.number_of_base_latent_features + self.config.number_of_pressure_bc_latent_features] = self.bc_transform_pressure(boundary_data[pressure_mask, :1])
        new_features[velocity_mask, self.config.number_of_base_latent_features:self.config.number_of_base_latent_features + self.config.number_of_velocities_bc_latent_features] = self.bc_transform_velocity(boundary_data[velocity_mask, :2])
        messages = self.message(x[edge_index[0]], x[edge_index[1]], pos[edge_index[0]], pos[edge_index[1]], node_type[edge_index[0]], node_type[edge_index[1]])
        aggr_out = self.aggregate(messages, edge_index[1], dim_size=graph_data.num_nodes)
        base_features = x[:, :self.config.number_of_base_latent_features]
        derivatives = torch.zeros_like(base_features)
        derivatives[node_type.squeeze(-1) == 0] = aggr_out[node_type.squeeze(-1) == 0]
        derivatives[pressure_mask] = self.bc_correction_pressure(base_features[pressure_mask], aggr_out[pressure_mask], new_features[pressure_mask, self.config.number_of_base_latent_features:self.config.number_of_base_latent_features + self.config.number_of_pressure_bc_latent_features])
        derivatives[velocity_mask] = self.bc_correction_velocity(base_features[velocity_mask], aggr_out[velocity_mask], new_features[velocity_mask, self.config.number_of_base_latent_features:self.config.number_of_base_latent_features + self.config.number_of_velocities_bc_latent_features])
        return derivatives