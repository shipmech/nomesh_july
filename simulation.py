# simulation.py
import torch
from lightning.pytorch import LightningModule
from torch.optim import Adam
from torch_geometric.data import Data
from model import DynamicsGNN, ICImprintingNN, BCTransformingNN, BCCorrectionNN
from mapping_gnn import MappingGNN
from integrator import TimeIntegrator
from loss import PhysicsLoss
from config import SimulationConfig
from output import save_vtk, log_case_params
from case import Case  # Assuming case.py is implemented

class FluidSimulation(LightningModule):
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.dynamics_gnn = DynamicsGNN(config)
        self.mapping_gnn = MappingGNN(config)
        self.ic_imprinting = ICImprintingNN(config)
        self.integrator = TimeIntegrator(config.dt, config.number_of_base_latent_features, config)
        self.physics_loss = PhysicsLoss(config)
        self.save_hyperparameters()

    def forward(self, graph_data: Data) -> Data:
        return self.dynamics_gnn(graph_data)

    def initialize_graph(self, case: Case) -> Data:
        graph_data = case.get_graph_data()
        initial_physical = graph_data.x[:, -6:-3]  # u, v, p
        base_features = self.ic_imprinting(initial_physical)
        graph_data.x[:, :self.config.number_of_base_latent_features] = base_features
        return graph_data

    def training_step(self, batch: Case, batch_idx: int) -> torch.Tensor:
        case = batch
        graph_data = self.initialize_graph(case)
        target_points = case.get_mesh().get_nodes()
        pos = graph_data.pos
        edge_index = graph_data.edge_index
        radius_tensor = torch.tensor(self.config.radius, device=self.device)
        pseudo = (pos[edge_index[1]] - pos[edge_index[0]]) / radius_tensor

        # Compute initial quantities after imprinting
        initial_quantities = self.mapping_gnn(graph_data, target_points, ensure_all_nodes=False).x[:, :3]  # Mapped u,v,p
        ground_truth_ic = case.initial_condition.get_initial_state().repeat(initial_quantities.shape[0], 1)

        quantities_list = [initial_quantities]
        derivatives_list = []
        prev_quantities = None
        losses = []

        num_steps = int(self.config.simulation_time / self.config.dt)
        for step in range(num_steps):
            t = step * self.config.dt
            case.update(t)

            # Compute derivatives
            derivatives = self.dynamics_gnn.compute_derivatives(graph_data)
            derivatives_list.append(derivatives)

            # Compute loss for this step (using mapped quantities)
            mapped_data = self.mapping_gnn(graph_data, target_points, ensure_all_nodes=False)
            quantities = mapped_data.x[:, :3]  # Assuming x has u,v,p first after mapping
            is_initial = (step == 0)
            loss, loss_dict = self.physics_loss(quantities, prev_quantities, derivatives, self.config.dt, pos, edge_index, pseudo, is_initial, ground_truth_ic if is_initial else None)
            losses.append(loss)

            # Save VTK
            if step % self.config.vtk_save_frequency == 0:
                save_vtk(graph_data, mapped_data, self.config.output_dir, self.current_epoch, str(batch_idx), step, self.config)

            # Step integrator
            aux_nns = {
                'bc_transform_pressure': self.dynamics_gnn.bc_transform_pressure,
                'bc_transform_velocity': self.dynamics_gnn.bc_transform_velocity,
                'bc_correction_pressure': self.dynamics_gnn.bc_correction_pressure,
                'bc_correction_velocity': self.dynamics_gnn.bc_correction_velocity
            }
            bc_values_list = [[bc.get_value(t) for bc in case.boundary_conditions]]
            graph_data = self.integrator.step(self.dynamics_gnn, graph_data, aux_nns, bc_values_list, t)

            prev_quantities = quantities

        total_loss = torch.mean(torch.stack(losses))
        self.log("train_loss", total_loss, prog_bar=True)

        if self.current_epoch % self.config.log_case_params_frequency == 0:
            log_case_params([case], self.config.output_dir, self.current_epoch)

        return total_loss

    def validation_step(self, batch: Case, batch_idx: int) -> torch.Tensor:
        case = batch
        graph_data = self.initialize_graph(case)
        target_points = case.get_mesh().get_nodes()
        pos = graph_data.pos
        edge_index = graph_data.edge_index
        radius_tensor = torch.tensor(self.config.radius, device=self.device)
        pseudo = (pos[edge_index[1]] - pos[edge_index[0]]) / radius_tensor

        initial_quantities = self.mapping_gnn(graph_data, target_points, ensure_all_nodes=False).x[:, :3]
        ground_truth_ic = case.initial_condition.get_initial_state().repeat(initial_quantities.shape[0], 1)

        quantities_list = [initial_quantities]
        derivatives_list = []
        prev_quantities = None
        losses = []

        num_steps = int(self.config.simulation_time / self.config.dt)
        for step in range(num_steps):
            t = step * self.config.dt
            case.update(t)

            derivatives = self.dynamics_gnn.compute_derivatives(graph_data)
            derivatives_list.append(derivatives)

            mapped_data = self.mapping_gnn(graph_data, target_points, ensure_all_nodes=False)
            quantities = mapped_data.x[:, :3]
            is_initial = (step == 0)
            loss, loss_dict = self.physics_loss(quantities, prev_quantities, derivatives, self.config.dt, pos, edge_index, pseudo, is_initial, ground_truth_ic if is_initial else None)
            losses.append(loss)

            if step % self.config.vtk_save_frequency == 0:
                save_vtk(graph_data, mapped_data, self.config.output_dir, self.current_epoch, f"val_{batch_idx}", step, self.config)

            aux_nns = {
                'bc_transform_pressure': self.dynamics_gnn.bc_transform_pressure,
                'bc_transform_velocity': self.dynamics_gnn.bc_transform_velocity,
                'bc_correction_pressure': self.dynamics_gnn.bc_correction_pressure,
                'bc_correction_velocity': self.dynamics_gnn.bc_correction_velocity
            }
            bc_values_list = [[bc.get_value(t) for bc in case.boundary_conditions]]
            graph_data = self.integrator.step(self.dynamics_gnn, graph_data, aux_nns, bc_values_list, t)

            prev_quantities = quantities

        val_loss = torch.mean(torch.stack(losses))
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer