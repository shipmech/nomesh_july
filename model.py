# model.py
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from config import SimulationConfig
from mesh import update_edges

class ICImprintingNN(nn.Module):
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(3, config.hidden_dim),  # Changed to 3 for u,v,p
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
            nn.Linear(config.number_of_base_latent_features + config.number_of_base_latent_features + bc_features_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.number_of_base_latent_features)
        )

    def forward(self, base_features: torch.Tensor, aggregated_messages: torch.Tensor, bc_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([base_features, aggregated_messages, bc_features], dim=-1)
        return self.mlp(combined)

class DynamicsGNN(MessagePassing):
    def __init__(self, config: SimulationConfig):
        super().__init__(aggr='add')
        self.config = config

        node_types = ['free', 'pressure', 'velocity']

        self.message_nns = nn.ModuleDict({
            f'{src}_{dst}': nn.Sequential(
                nn.Linear(2 * config.number_of_base_latent_features + 1, config.hidden_dim),  # x_j || x_i || dist
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.number_of_base_latent_features)
            ) for src in node_types for dst in node_types
        })

        self.to_physic_transformation_nns = nn.ModuleDict({
            typ: nn.Sequential(
                nn.Linear(config.number_of_base_latent_features, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 6)  # u,v,p,u_t,v_t,p_t
            ) for typ in node_types
        })

        self.bc_transform_pressure = BCTransformingNN(1, config.number_of_pressure_bc_latent_features, config)  # pressure BC
        self.bc_transform_velocity = BCTransformingNN(2, config.number_of_velocities_bc_latent_features, config)  # velocity BC (u,v)

        self.bc_correction_pressure = BCCorrectionNN(config, config.number_of_pressure_bc_latent_features)
        self.bc_correction_velocity = BCCorrectionNN(config, config.number_of_velocities_bc_latent_features)

    def forward(self, graph_data: Data) -> Data:
        graph_data.edge_index = update_edges(graph_data.pos, self.config.k_neighbors, self.config.radius)
        out = self.propagate(edge_index=graph_data.edge_index, x=graph_data.x, pos=graph_data.pos, node_type=graph_data.node_type)
        graph_data.x[:, :self.config.number_of_base_latent_features] = out
        return graph_data

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, pos_j: torch.Tensor, pos_i: torch.Tensor, node_type_i: torch.Tensor, node_type_j: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)
        combined = torch.cat([x_j, x_i, dist], dim=-1)
        src_type = node_type_j.squeeze(-1).long()
        dst_type = node_type_i.squeeze(-1).long()
        messages = torch.zeros(combined.shape[0], self.config.number_of_base_latent_features, device=combined.device)
        for s in range(3):
            for d in range(3):
                mask = (src_type == s) & (dst_type == d)
                if mask.sum() > 0:
                    key = f'{["free", "pressure", "velocity"][s]}_{["free", "pressure", "velocity"][d]}'
                    messages[mask] = self.message_nns[key](combined[mask])
        return messages

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        base_features = x[:, :self.config.number_of_base_latent_features]
        bc_features = x[:, self.config.number_of_base_latent_features : self.config.number_of_base_latent_features + max(self.config.number_of_pressure_bc_latent_features, self.config.number_of_velocities_bc_latent_features)]
        phys_features = x[:, -6:]

        mask_free = (node_type.squeeze(-1) == 0)
        mask_pressure = (node_type.squeeze(-1) == 1)
        mask_velocity = (node_type.squeeze(-1) == 2)

        updated_features = aggr_out.clone()
        if mask_pressure.sum() > 0:
            updated_features[mask_pressure] = self.bc_correction_pressure(base_features[mask_pressure], aggr_out[mask_pressure], bc_features[mask_pressure])
        if mask_velocity.sum() > 0:
            updated_features[mask_velocity] = self.bc_correction_velocity(base_features[mask_velocity], aggr_out[mask_velocity], bc_features[mask_velocity])

        # Compute physical quantities
        phys_out = torch.zeros_like(phys_features)
        for typ, mask in zip(['free', 'pressure', 'velocity'], [mask_free, mask_pressure, mask_velocity]):
            if mask.sum() > 0:
                phys_out[mask] = self.to_physic_transformation_nns[typ](updated_features[mask])

        x[:, :self.config.number_of_base_latent_features] = updated_features
        x[:, -6:] = phys_out

        return updated_features  # For MessagePassing

    def compute_derivatives(self, graph_data: Data) -> torch.Tensor:
        # Assuming derivatives are part of phys_out, e.g., u_t, v_t, p_t
        return graph_data.x[:, -3:]  # u_t, v_t, p_t