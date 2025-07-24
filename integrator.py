import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from mesh import update_edges

class TimeIntegrator(nn.Module):
    def __init__(self, dt, number_of_base_latent_features, config):
        super().__init__()
        self.dt = dt
        self.base_features_dim = number_of_base_latent_features
        self.config = config

    def compute_derivatives(self, gnn, data, aux_nns, bc_values, t):
        assert data.pos.dim() == 2, f"Expected data.pos to be 2D, got shape {data.pos.shape}"
        data_clone = Data(
            pos=data.pos,
            x=data.x.clone(),
            edge_index=data.edge_index,
            node_type=data.node_type,
            boundary_data=bc_values,
            width=data.width if hasattr(data, 'width') else None,
            height=data.height if hasattr(data, 'height') else None
        )
        data_clone = update_edges(data_clone, self.config)
        derivatives = aux_nns['dynamics_gnn'].compute_derivatives(data_clone)
        return derivatives

    def step(self, gnn, batch_data, aux_nns, bc_values_list, t):
        if isinstance(batch_data, Batch):
            data_list = batch_data.to_data_list()
        else:
            data_list = [batch_data]
        
        if len(data_list) != len(bc_values_list):
            raise ValueError(f"Number of graphs ({len(data_list)}) does not match number of bc_values ({len(bc_values_list)})")
        
        updated_data_list = []
        for i, single_data in enumerate(data_list):
            bc_values = bc_values_list[i]
            single_data = update_edges(single_data, self.config)
            k1 = self.compute_derivatives(gnn, single_data, aux_nns, bc_values, t)
            k2_data = Data(
                pos=single_data.pos,
                x=single_data.x.clone(),
                edge_index=single_data.edge_index,
                node_type=single_data.node_type,
                boundary_data=bc_values,
                width=single_data.width if hasattr(single_data, 'width') else None,
                height=single_data.height if hasattr(single_data, 'height') else None
            )
            k2_data.x[:, :self.base_features_dim] = single_data.x[:, :self.base_features_dim] + 0.5 * self.dt * k1
            k2_data = update_edges(k2_data, self.config)
            k2 = self.compute_derivatives(gnn, k2_data, aux_nns, bc_values, t + 0.5 * self.dt)
            k3_data = Data(
                pos=single_data.pos,
                x=single_data.x.clone(),
                edge_index=single_data.edge_index,
                node_type=single_data.node_type,
                boundary_data=bc_values,
                width=single_data.width if hasattr(single_data, 'width') else None,
                height=single_data.height if hasattr(single_data, 'height') else None
            )
            k3_data.x[:, :self.base_features_dim] = single_data.x[:, :self.base_features_dim] + 0.5 * self.dt * k2
            k3_data = update_edges(k3_data, self.config)
            k3 = self.compute_derivatives(gnn, k3_data, aux_nns, bc_values, t + 0.5 * self.dt)
            k4_data = Data(
                pos=single_data.pos,
                x=single_data.x.clone(),
                edge_index=single_data.edge_index,
                node_type=single_data.node_type,
                boundary_data=bc_values,
                width=single_data.width if hasattr(single_data, 'width') else None,
                height=single_data.height if hasattr(single_data, 'height') else None
            )
            k4_data.x[:, :self.base_features_dim] = single_data.x[:, :self.base_features_dim] + self.dt * k3
            k4_data = update_edges(k4_data, self.config)
            k4 = self.compute_derivatives(gnn, k4_data, aux_nns, bc_values, t + self.dt)
            single_data.x[:, :self.base_features_dim] += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            updated_data_list.append(single_data)
        
        return Batch.from_data_list(updated_data_list)