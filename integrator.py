# integrator.py
import torch
from torch_geometric.data import Data
from config import SimulationConfig
from model import DynamicsGNN
from mesh import update_edges

class TimeIntegrator:
    def __init__(self, dt: float, config: SimulationConfig):
        self.dt = dt
        self.config = config

    def compute_derivatives(self, gnn: DynamicsGNN, data: Data, aux_nns: dict[str, torch.nn.Module], bc_values: list[torch.Tensor], t: float) -> torch.Tensor:
        # Clone data to avoid in-place modification
        data = data.clone()
        data.x = data.x.clone()
        
        # Apply boundary transformations
        bc_transform_velocity = aux_nns['bc_transform_velocity']
        bc_transform_pressure = aux_nns['bc_transform_pressure']

        # bc_values[0]: velocity [u, v], shape (2,)
        # bc_values[1]: pressure [p], shape (1,)
        velocity_bc = bc_transform_velocity(bc_values[0].unsqueeze(0))  # (1, number_of_velocities_bc_latent_features)
        pressure_bc = bc_transform_pressure(bc_values[1].unsqueeze(0))  # (1, number_of_pressure_bc_latent_features)

        # Update data.x with BC features
        mask_pressure = data.node_type.squeeze(-1) == 1
        mask_velocity = data.node_type.squeeze(-1) == 2

        bc_start = self.config.number_of_base_latent_features
        if mask_pressure.sum() > 0:
            data.x[mask_pressure, bc_start:bc_start + self.config.number_of_pressure_bc_latent_features] = pressure_bc.repeat(mask_pressure.sum(), 1)
        if mask_velocity.sum() > 0:
            data.x[mask_velocity, bc_start:bc_start + self.config.number_of_velocities_bc_latent_features] = velocity_bc.repeat(mask_velocity.sum(), 1)

        # Forward pass to compute derivatives
        gnn(data)
        return data.x[:, -3:]  # u_t, v_t, p_t

    def step(self, gnn: DynamicsGNN, batch_data: Data, aux_nns: dict[str, torch.nn.Module], bc_values_list: list[list[torch.Tensor]], t: float) -> Data:
        # Assuming single graph for simplicity (batch_size=1)
        data_list = [batch_data]

        # RK4 steps
        for i, single_data in enumerate(data_list):
            single_data.edge_index = update_edges(single_data.pos, self.config.k_neighbors, self.config.radius)

            bc_values = bc_values_list[i]

            k1 = self.compute_derivatives(gnn, single_data, aux_nns, bc_values, t)
            temp_data = single_data.clone()
            temp_data.x = temp_data.x.clone()
            left = temp_data.x[:, :-6]
            mid = temp_data.x[:, -6:-3] + 0.5 * self.dt * k1
            right = temp_data.x[:, -3:]
            temp_data.x = torch.cat([left, mid, right], dim=-1)
            k2 = self.compute_derivatives(gnn, temp_data, aux_nns, bc_values, t + 0.5 * self.dt)

            temp_data = single_data.clone()
            temp_data.x = temp_data.x.clone()
            left = temp_data.x[:, :-6]
            mid = temp_data.x[:, -6:-3] + 0.5 * self.dt * k2
            right = temp_data.x[:, -3:]
            temp_data.x = torch.cat([left, mid, right], dim=-1)
            k3 = self.compute_derivatives(gnn, temp_data, aux_nns, bc_values, t + 0.5 * self.dt)

            temp_data = single_data.clone()
            temp_data.x = temp_data.x.clone()
            left = temp_data.x[:, :-6]
            mid = temp_data.x[:, -6:-3] + self.dt * k3
            right = temp_data.x[:, -3:]
            temp_data.x = torch.cat([left, mid, right], dim=-1)
            k4 = self.compute_derivatives(gnn, temp_data, aux_nns, bc_values, t + self.dt)

            update = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            left = single_data.x[:, :-6]
            mid = single_data.x[:, -6:-3] + self.dt * update
            right = single_data.x[:, -3:]
            single_data.x = torch.cat([left, mid, right], dim=-1)

            data_list[i] = single_data

        return data_list[0]  # Return single graph