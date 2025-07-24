# integrator.py
import torch
from torch_geometric.data import Batch, Data
from config import SimulationConfig
from model import DynamicsGNN
from mesh import update_edges

class TimeIntegrator:
    def __init__(self, dt: float, base_features_dim: int, config: SimulationConfig):
        self.dt = dt
        self.base_features_dim = base_features_dim
        self.config = config

    def compute_derivatives(self, gnn: DynamicsGNN, data: Data, aux_nns: dict[str, torch.nn.Module], bc_values: torch.Tensor, t: float) -> torch.Tensor:
        # Apply boundary transformations
        bc_transform_pressure = aux_nns['bc_transform_pressure']
        bc_transform_velocity = aux_nns['bc_transform_velocity']

        # Assuming bc_values has shape for pressure and velocity
        pressure_bc = bc_transform_pressure(bc_values[0:1])  # Example slicing
        velocity_bc = bc_transform_velocity(bc_values[1:])  # Example

        # Update data.x with BC features (simplified)
        mask_pressure = data.node_type == 1
        mask_velocity = data.node_type == 2
        data.x[mask_pressure, gnn.config.number_of_base_latent_features:gnn.config.number_of_base_latent_features + gnn.config.number_of_pressure_bc_latent_features] = pressure_bc.repeat(mask_pressure.sum(), 1)
        data.x[mask_velocity, gnn.config.number_of_base_latent_features:gnn.config.number_of_base_latent_features + gnn.config.number_of_velocities_bc_latent_features] = velocity_bc.repeat(mask_velocity.sum(), 1)

        # Forward pass to compute derivatives
        gnn(data)
        return data.x[:, -3:]  # u_t, v_t, p_t

    def step(self, gnn: DynamicsGNN, batch_data: Data, aux_nns: dict[str, torch.nn.Module], bc_values_list: list[list[torch.Tensor]], t: float) -> Data:
        # Assuming single graph for simplicity (batch_size=1)
        data_list = [batch_data]

        # RK4 steps
        for i, single_data in enumerate(data_list):
            single_data = update_edges(single_data.pos, self.config.k_neighbors, self.config.radius)

            bc_values = bc_values_list[i]

            k1 = self.compute_derivatives(gnn, single_data.clone(), aux_nns, bc_values, t)
            temp_data = single_data.clone()
            temp_data.x[:, :self.base_features_dim] += 0.5 * self.dt * k1
            k2 = self.compute_derivatives(gnn, temp_data, aux_nns, bc_values, t + 0.5 * self.dt)

            temp_data = single_data.clone()
            temp_data.x[:, :self.base_features_dim] += 0.5 * self.dt * k2
            k3 = self.compute_derivatives(gnn, temp_data, aux_nns, bc_values, t + 0.5 * self.dt)

            temp_data = single_data.clone()
            temp_data.x[:, :self.base_features_dim] += self.dt * k3
            k4 = self.compute_derivatives(gnn, temp_data, aux_nns, bc_values, t + self.dt)

            update = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            single_data.x[:, :self.base_features_dim] += self.dt * update

            data_list[i] = single_data

        return data_list[0]  # Return single graph