import torch
import lightning as L
from torch_geometric.data import Data, Batch
from config import SimulationConfig
from model import DynamicsGNN, ICImprintingNN
from mapping_gnn import MappingGNN
from integrator import TimeIntegrator as RK4Integrator
from loss import PhysicsLoss
from output import save_vtk, log_case_params
from mesh import create_meshes

class FluidSimulation(L.LightningModule):
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.save_hyperparameters()  # Save config as hyperparameters
        self.config = config
        self.dynamics_gnn = DynamicsGNN(config)
        self.mapping_gnn = MappingGNN(config)
        self.ic_imprinting = ICImprintingNN(config)
        self.integrator = RK4Integrator(config.dt, config.number_of_base_latent_features, config)
        self.physics_loss = PhysicsLoss(config)

    def initialize_graph(self, case: Data):
        graph_data, target_points = create_meshes(case, self.config)
        initial_quantities = torch.zeros(graph_data.num_nodes, 6, device=self.device)
        graph_data.x[:, :self.config.number_of_base_latent_features] = self.ic_imprinting(initial_quantities)
        return graph_data, target_points

    def step(self, graph_data: Data, target_points: torch.Tensor, bc_values: torch.Tensor, num_steps: int):
        aux_nns = {
            'dynamics_gnn': self.dynamics_gnn,
            'bc_transform_pressure': self.dynamics_gnn.bc_transform_pressure,
            'bc_transform_velocity': self.dynamics_gnn.bc_transform_velocity,
            'bc_correction_pressure': self.dynamics_gnn.bc_correction_pressure,
            'bc_correction_velocity': self.dynamics_gnn.bc_correction_velocity
        }
        mesh_data_list = []
        for step in range(num_steps):
            graph_data = self.dynamics_gnn.forward(graph_data)
            batch_graph = Batch.from_data_list([graph_data])
            batch_graph = self.integrator.step(
                self.dynamics_gnn,
                batch_graph,
                aux_nns=aux_nns,
                bc_values_list=[bc_values],
                t=step * self.config.dt
            )
            graph_data = batch_graph.to_data_list()[0]
            # Ensure all graph_data nodes are mapped
            mesh_data = self.mapping_gnn(graph_data, target_points, ensure_all_nodes=True)
            print(f"Step {step}: graph_data.num_nodes={graph_data.num_nodes}, mesh_data.num_nodes={mesh_data.num_nodes}")
            mesh_data_list.append(mesh_data)
            if step % self.config.vtk_save_frequency == 0:
                save_vtk(graph_data, mesh_data, self.config.output_dir, self.current_epoch, graph_data.case_id, step, self.config)
        return graph_data, mesh_data_list

    def training_step(self, batch, batch_idx):
        case = batch  # DataLoader returns a single Data object due to batch_size=1
        graph_data, target_points = self.initialize_graph(case)
        num_steps = int(self.config.simulation_time / self.config.dt)

        graph_data, mesh_data_list = self.step(graph_data, target_points, case.boundary_data, num_steps)
        
        if self.global_step % self.config.log_case_params_frequency == 0:
            log_case_params([case], self.config.output_dir, self.current_epoch)

        quantities = torch.stack([m.x for m in mesh_data_list])
        derivatives = torch.stack([m.derivatives for m in mesh_data_list])
        prev_quantities = torch.cat([torch.zeros(1, quantities.shape[1], 6, device=self.device), quantities[:-1]], dim=0)

        # Physics loss for all nodes over the entire domain
        physics_loss, physics_component_losses = self.physics_loss.compute_loss(quantities, prev_quantities, derivatives, self.config.dt, target_points)

        # Boundary condition loss
        bc_loss = torch.tensor(0.0, device=self.device)
        if quantities.size(0) > 0:  # Ensure mesh_data_list is not empty
            last_graph_quantities = graph_data.x[:, -6:]  # Use graph data for the last time step
            boundary_mask = (graph_data.node_type == 2) | (graph_data.node_type == 1)  # All boundary nodes
            inlet_mask = (graph_data.node_type == 2) & torch.isclose(graph_data.pos[:, 0], torch.tensor(0.0, dtype=torch.float32, device=self.device), atol=0.005)
            outlet_mask = (graph_data.node_type == 1) & torch.isclose(graph_data.pos[:, 0], torch.tensor(graph_data.width, dtype=torch.float32, device=self.device), atol=0.005)
            top_bottom_mask = (graph_data.node_type == 2) & (
                torch.isclose(graph_data.pos[:, 1], torch.tensor(0.0, dtype=torch.float32, device=self.device), atol=0.005) |
                torch.isclose(graph_data.pos[:, 1], torch.tensor(graph_data.height, dtype=torch.float32, device=self.device), atol=0.005)
            )

            print(f"Training - boundary_mask sum: {boundary_mask.sum()}, graph_data.num_nodes: {graph_data.num_nodes}")
            if boundary_mask.any():
                bc_diff = torch.zeros_like(last_graph_quantities[boundary_mask, :2], device=self.device)
                bc_expected = case.boundary_data[boundary_mask, :2].to(self.device)
                boundary_indices = boundary_mask.nonzero(as_tuple=True)[0].to(self.device)
                boundary_size = boundary_mask.sum().item()

                # Handle inlet nodes (velocity)
                if inlet_mask.any():
                    inlet_indices_full = inlet_mask.nonzero(as_tuple=True)[0].to(self.device)
                    inlet_indices_boundary = boundary_indices[inlet_mask[boundary_mask]]
                    inlet_subset_indices = torch.arange(boundary_size, device=self.device)[inlet_mask[boundary_mask]]
                    if inlet_indices_boundary.numel() > 0 and inlet_indices_boundary.max() < graph_data.num_nodes:
                        if inlet_subset_indices.numel() == inlet_indices_boundary.numel():
                            inlet_diff = last_graph_quantities[inlet_indices_boundary, :2] - bc_expected[inlet_subset_indices, :2]
                            if inlet_diff.size(0) == inlet_subset_indices.numel():
                                bc_diff[inlet_subset_indices] = inlet_diff  # Use subset indices for bc_diff
                            else:
                                print(f"Warning: inlet_diff shape {inlet_diff.shape} mismatch with inlet_subset_indices {inlet_subset_indices.numel()}")
                        else:
                            print(f"Warning: inlet_subset_indices mismatch: {inlet_subset_indices.numel()} vs {inlet_indices_boundary.numel()}")
                    else:
                        print(f"Warning: Invalid inlet_indices in training_step: {inlet_indices_boundary}")

                # Handle outlet nodes (pressure)
                if outlet_mask.any():
                    outlet_indices_full = outlet_mask.nonzero(as_tuple=True)[0].to(self.device)
                    outlet_indices_boundary = boundary_indices[outlet_mask[boundary_mask]]
                    outlet_subset_indices = torch.arange(boundary_size, device=self.device)[outlet_mask[boundary_mask]]
                    if outlet_indices_boundary.numel() > 0 and outlet_indices_boundary.max() < graph_data.num_nodes:
                        if outlet_subset_indices.numel() == outlet_indices_boundary.numel():
                            outlet_diff = last_graph_quantities[outlet_indices_boundary, 2:3] - bc_expected[outlet_subset_indices, 0:1]
                            if outlet_diff.size(0) == outlet_subset_indices.numel():
                                bc_diff[outlet_subset_indices, 0] = outlet_diff.squeeze(-1)  # Use subset indices for bc_diff
                            else:
                                print(f"Warning: outlet_diff shape {outlet_diff.shape} mismatch with outlet_subset_indices {outlet_subset_indices.numel()}")
                        else:
                            print(f"Warning: outlet_subset_indices mismatch: {outlet_subset_indices.numel()} vs {outlet_indices_boundary.numel()}")
                    else:
                        print(f"Warning: Invalid outlet_indices in training_step: {outlet_indices_boundary}")

                # Handle top/bottom nodes (velocity)
                if top_bottom_mask.any():
                    top_bottom_indices_full = top_bottom_mask.nonzero(as_tuple=True)[0].to(self.device)
                    top_bottom_indices_boundary = boundary_indices[top_bottom_mask[boundary_mask]]
                    top_bottom_subset_indices = torch.arange(boundary_size, device=self.device)[top_bottom_mask[boundary_mask]]
                    if top_bottom_indices_boundary.numel() > 0 and top_bottom_indices_boundary.max() < graph_data.num_nodes:
                        if top_bottom_subset_indices.numel() == top_bottom_indices_boundary.numel():
                            top_bottom_diff = last_graph_quantities[top_bottom_indices_boundary, :2] - bc_expected[top_bottom_subset_indices, :2]
                            if top_bottom_diff.size(0) == top_bottom_subset_indices.numel():
                                bc_diff[top_bottom_subset_indices] = top_bottom_diff  # Use subset indices for bc_diff
                            else:
                                print(f"Warning: top_bottom_diff shape {top_bottom_diff.shape} mismatch with top_bottom_subset_indices {top_bottom_subset_indices.numel()}")
                        else:
                            print(f"Warning: top_bottom_subset_indices mismatch: {top_bottom_subset_indices.numel()} vs {top_bottom_indices_boundary.numel()}")
                    else:
                        print(f"Warning: Invalid top_bottom_indices in training_step: {top_bottom_indices_boundary}")

                bc_loss = torch.mean(bc_diff ** 2) * 1e4  # Scale similar to physics loss

        # Total loss
        total_loss = physics_loss + bc_loss
        self.log('train_loss', total_loss, prog_bar=True, batch_size=1)
        self.log('train_physics_loss', physics_loss, batch_size=1)
        self.log('train_bc_loss', bc_loss, batch_size=1)
        for i, cl in enumerate(physics_component_losses):
            self.log(f'train_component_loss_{i}', cl, batch_size=1)

        return total_loss

    def validation_step(self, batch, batch_idx):
        case = batch
        graph_data, target_points = self.initialize_graph(case)
        num_steps = int(self.config.simulation_time / self.config.dt)

        graph_data, mesh_data_list = self.step(graph_data, target_points, case.boundary_data, num_steps)

        quantities = torch.stack([m.x for m in mesh_data_list])
        derivatives = torch.stack([m.derivatives for m in mesh_data_list])
        prev_quantities = torch.cat([torch.zeros(1, quantities.shape[1], 6, device=self.device), quantities[:-1]], dim=0)

        # Physics loss for all nodes over the entire domain
        physics_loss, physics_component_losses = self.physics_loss.compute_loss(quantities, prev_quantities, derivatives, self.config.dt, target_points)

        # Boundary condition loss
        bc_loss = torch.tensor(0.0, device=self.device)
        if quantities.size(0) > 0:  # Ensure mesh_data_list is not empty
            last_graph_quantities = graph_data.x[:, -6:]  # Use graph data for the last time step
            boundary_mask = (graph_data.node_type == 2) | (graph_data.node_type == 1)  # All boundary nodes
            inlet_mask = (graph_data.node_type == 2) & torch.isclose(graph_data.pos[:, 0], torch.tensor(0.0, dtype=torch.float32, device=self.device), atol=0.005)
            outlet_mask = (graph_data.node_type == 1) & torch.isclose(graph_data.pos[:, 0], torch.tensor(graph_data.width, dtype=torch.float32, device=self.device), atol=0.005)
            top_bottom_mask = (graph_data.node_type == 2) & (
                torch.isclose(graph_data.pos[:, 1], torch.tensor(0.0, dtype=torch.float32, device=self.device), atol=0.005) |
                torch.isclose(graph_data.pos[:, 1], torch.tensor(graph_data.height, dtype=torch.float32, device=self.device), atol=0.005)
            )

            print(f"Validation - boundary_mask sum: {boundary_mask.sum()}, graph_data.num_nodes: {graph_data.num_nodes}")
            if boundary_mask.any():
                bc_diff = torch.zeros_like(last_graph_quantities[boundary_mask, :2], device=self.device)
                bc_expected = case.boundary_data[boundary_mask, :2].to(self.device)
                boundary_indices = boundary_mask.nonzero(as_tuple=True)[0].to(self.device)
                boundary_size = boundary_mask.sum().item()

                # Handle inlet nodes (velocity)
                if inlet_mask.any():
                    inlet_indices_full = inlet_mask.nonzero(as_tuple=True)[0].to(self.device)
                    inlet_indices_boundary = boundary_indices[inlet_mask[boundary_mask]]
                    inlet_subset_indices = torch.arange(boundary_size, device=self.device)[inlet_mask[boundary_mask]]
                    if inlet_indices_boundary.numel() > 0 and inlet_indices_boundary.max() < graph_data.num_nodes:
                        if inlet_subset_indices.numel() == inlet_indices_boundary.numel():
                            inlet_diff = last_graph_quantities[inlet_indices_boundary, :2] - bc_expected[inlet_subset_indices, :2]
                            if inlet_diff.size(0) == inlet_subset_indices.numel():
                                bc_diff[inlet_subset_indices] = inlet_diff  # Use subset indices for bc_diff
                            else:
                                print(f"Warning: inlet_diff shape {inlet_diff.shape} mismatch with inlet_subset_indices {inlet_subset_indices.numel()}")
                        else:
                            print(f"Warning: inlet_subset_indices mismatch: {inlet_subset_indices.numel()} vs {inlet_indices_boundary.numel()}")
                    else:
                        print(f"Warning: Invalid inlet_indices in validation_step: {inlet_indices_boundary}")

                # Handle outlet nodes (pressure)
                if outlet_mask.any():
                    outlet_indices_full = outlet_mask.nonzero(as_tuple=True)[0].to(self.device)
                    outlet_indices_boundary = boundary_indices[outlet_mask[boundary_mask]]
                    outlet_subset_indices = torch.arange(boundary_size, device=self.device)[outlet_mask[boundary_mask]]
                    if outlet_indices_boundary.numel() > 0 and outlet_indices_boundary.max() < graph_data.num_nodes:
                        if outlet_subset_indices.numel() == outlet_indices_boundary.numel():
                            outlet_diff = last_graph_quantities[outlet_indices_boundary, 2:3] - bc_expected[outlet_subset_indices, 0:1]
                            if outlet_diff.size(0) == outlet_subset_indices.numel():
                                bc_diff[outlet_subset_indices, 0] = outlet_diff.squeeze(-1)  # Use subset indices for bc_diff
                            else:
                                print(f"Warning: outlet_diff shape {outlet_diff.shape} mismatch with outlet_subset_indices {outlet_subset_indices.numel()}")
                        else:
                            print(f"Warning: outlet_subset_indices mismatch: {outlet_subset_indices.numel()} vs {outlet_indices_boundary.numel()}")
                    else:
                        print(f"Warning: Invalid outlet_indices in validation_step: {outlet_indices_boundary}")

                # Handle top/bottom nodes (velocity)
                if top_bottom_mask.any():
                    top_bottom_indices_full = top_bottom_mask.nonzero(as_tuple=True)[0].to(self.device)
                    top_bottom_indices_boundary = boundary_indices[top_bottom_mask[boundary_mask]]
                    top_bottom_subset_indices = torch.arange(boundary_size, device=self.device)[top_bottom_mask[boundary_mask]]
                    if top_bottom_indices_boundary.numel() > 0 and top_bottom_indices_boundary.max() < graph_data.num_nodes:
                        if top_bottom_subset_indices.numel() == top_bottom_indices_boundary.numel():
                            top_bottom_diff = last_graph_quantities[top_bottom_indices_boundary, :2] - bc_expected[top_bottom_subset_indices, :2]
                            if top_bottom_diff.size(0) == top_bottom_subset_indices.numel():
                                bc_diff[top_bottom_subset_indices] = top_bottom_diff  # Use subset indices for bc_diff
                            else:
                                print(f"Warning: top_bottom_diff shape {top_bottom_diff.shape} mismatch with top_bottom_subset_indices {top_bottom_subset_indices.numel()}")
                        else:
                            print(f"Warning: top_bottom_subset_indices mismatch: {top_bottom_subset_indices.numel()} vs {top_bottom_indices_boundary.numel()}")
                    else:
                        print(f"Warning: Invalid top_bottom_indices in validation_step: {top_bottom_indices_boundary}")

                bc_loss = torch.mean(bc_diff ** 2) * 1e4  # Scale similar to physics loss

        # Total loss
        total_loss = physics_loss + bc_loss
        self.log('val_loss', total_loss, prog_bar=True, batch_size=1, on_step=False, on_epoch=True)
        self.log('val_physics_loss', physics_loss, batch_size=1, on_step=False, on_epoch=True)
        self.log('val_bc_loss', bc_loss, batch_size=1, on_step=False, on_epoch=True)
        for i, cl in enumerate(physics_component_losses):
            self.log(f'val_component_loss_{i}', cl, batch_size=1, on_step=False, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.dynamics_gnn.parameters()) +
            list(self.mapping_gnn.parameters()) +
            list(self.ic_imprinting.parameters()),
            lr=self.config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }