# simulation.py
import torch
from lightning.pytorch import LightningModule
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import HeteroConv

from a_config import SimulationConfig
from bb_nn_models import BCTransformingMLP, TransformConv, MessagePassingMLPConv
from cc_output import save_graph_vtk, save_background_vtk, log_case_params
from d_case import Case
from e_model import Model

class FluidSimulation(LightningModule):
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.viscosity = config.viscosity
        self.density = config.density

        # Shared NNs as learnable modules
        self.bc_transform_pressure = BCTransformingMLP(1, config.number_of_pressure_bc_latent_features, config.number_bc_nn_hidden_dim)
        self.bc_transform_velocity = BCTransformingMLP(2, config.number_of_velocities_bc_latent_features, config.number_bc_nn_hidden_dim)

        num_base_f = config.number_of_base_latent_features
        physical_channels = config.num_phys_features + config.num_phys_spatial_features_d + config.num_phys_spatial_features_dd
        physical_dt_channels = config.num_phys_features
        dim = 2
        kernel_size = config.kernel_size_transfer

        self.transform_conv = TransformConv(num_base_f, physical_channels, dim=dim, kernel_size=kernel_size)
        self.transform_conv_dt = TransformConv(num_base_f, physical_dt_channels, dim=dim, kernel_size=kernel_size)
        self.reverse_transform_conv = TransformConv(physical_channels, num_base_f, dim=dim, kernel_size=kernel_size)

        node_types = ['Free', 'Press', 'NoSlip']
        bc_dims = {
            'Free': 0,
            'Press': config.number_of_pressure_bc_latent_features,
            'NoSlip': config.number_of_velocities_bc_latent_features,
        }
        conv_dict = {}
        for src in node_types:
            src_channels = num_base_f + bc_dims[src]
            out_channels = num_base_f
            for dst in node_types:
                edge_type = (src, 'influences', dst)
                conv_dict[edge_type] = MessagePassingMLPConv(src_channels, out_channels, hidden_dim=config.message_passing_hidden_dim)
        self.message_passing_conv = HeteroConv(conv_dict, aggr='sum')

        self.save_hyperparameters()

    def compute_physics_loss(self, quantities: torch.Tensor, derivatives: torch.Tensor, gradients: torch.Tensor, second_gradients: torch.Tensor) -> torch.Tensor:
        u, v, p = quantities[:, 0], quantities[:, 1], quantities[:, 2]
        u_t, v_t, p_t = derivatives[:, 0], derivatives[:, 1], derivatives[:, 2]

        u_x, u_y, v_x, v_y, p_x, p_y = gradients[:, 0], gradients[:, 1], gradients[:, 2], gradients[:, 3], gradients[:, 4], gradients[:, 5]
        u_xx, u_yy, v_xx, v_yy = second_gradients[:, 0], second_gradients[:, 1], second_gradients[:, 2], second_gradients[:, 3]

        # Continuity equation (incompressible): div(u) = u_x + v_y = 0
        continuity = u_x + v_y

        # Momentum equations (Navier-Stokes)
        momentum_x = u_t + u * u_x + v * u_y + (1 / self.density) * p_x - self.viscosity * (u_xx + u_yy)
        momentum_y = v_t + u * v_x + v * v_y + (1 / self.density) * p_y - self.viscosity * (v_xx + v_yy)

        # MSE losses
        loss_continuity = torch.mean(continuity ** 2)
        loss_momentum_x = torch.mean(momentum_x ** 2)
        loss_momentum_y = torch.mean(momentum_y ** 2)

        return loss_continuity + loss_momentum_x + loss_momentum_y

    def compute_ic_loss(self, quantities: torch.Tensor, ground_truth_ic: torch.Tensor) -> torch.Tensor:
        return torch.mean((quantities - ground_truth_ic) ** 2)
    
    def compute_bc_loss(self, model: Model) -> torch.Tensor:
        data = model.background_mesh.graph_hetero_data
        bc_loss = torch.tensor(0.0, device=self.device)

        if 'Press' in data.node_types and data['Press'].num_nodes > 0:
            pred_p = data['Press'].phys_features[:, 2]  # Predicted pressure
            target_p = data['Press'].bc_features.squeeze(-1) if data['Press'].bc_features.dim() > 1 else data['Press'].bc_features  # BC pressure (handle [N,1] or [N])
            bc_loss += torch.mean((pred_p - target_p) ** 2)

        if 'NoSlip' in data.node_types and data['NoSlip'].num_nodes > 0:
            pred_uv = data['NoSlip'].phys_features[:, 0:2]  # Predicted [u, v]
            target_uv = data['NoSlip'].bc_features  # BC [u, v] [N,2]
            bc_loss += torch.mean((pred_uv - target_uv) ** 2)

        return bc_loss

    def collect_background_features(self, model: Model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = model.background_mesh.graph_hetero_data
        node_types = data.node_types
        num_nodes = sum(data[t].pos.shape[0] for t in node_types)

        quantities = torch.zeros((num_nodes, self.config.num_phys_features), device=self.device)
        derivatives = torch.zeros((num_nodes, self.config.num_phys_features), device=self.device)
        gradients = torch.zeros((num_nodes, self.config.num_phys_spatial_features_d), device=self.device)
        second_gradients = torch.zeros((num_nodes, self.config.num_phys_spatial_features_dd), device=self.device)

        offset = 0
        for t in node_types:
            n = data[t].pos.shape[0]
            s = slice(offset, offset + n)
            quantities[s] = data[t].phys_features
            derivatives[s] = data[t].phys_features_dt
            gradients[s] = data[t].phys_features_spatial_d
            second_gradients[s] = data[t].phys_features_spatial_dd
            offset += n

        return quantities, derivatives, gradients, second_gradients

    def training_step(self, case: Case, batch_idx: int) -> torch.Tensor:
        model = Model(case)

        max_time = self.config.simulation_time
        delta_t = self.config.dt
        n_time_step = int(max_time / delta_t) + 1
        times = torch.linspace(0, max_time, n_time_step, device=self.device)

        current_time = 0.0

        model.update_BC(current_time, self.bc_transform_pressure, self.bc_transform_velocity)
        model.compute_base_features_dt(self.message_passing_conv)
        model.transfer_latent_to_physics(self.transform_conv, self.transform_conv_dt)

        quantities, derivatives, gradients, second_gradients = self.collect_background_features(model)

        # Assuming ground_truth_ic is zeros since not provided in case
        ground_truth_ic = torch.zeros_like(quantities)
        physics_loss = self.compute_physics_loss(quantities, derivatives, gradients, second_gradients)
        ic_loss = self.compute_ic_loss(quantities, ground_truth_ic)
        loss = physics_loss + self.config.lambda_ic * ic_loss
        losses = [loss]

        for step in range(n_time_step):
            current_time = times[step]

            model.next_time_step(current_time, delta_t, self.bc_transform_pressure, self.bc_transform_velocity, self.message_passing_conv)

            model.transfer_latent_to_physics(self.transform_conv, self.transform_conv_dt)

            quantities, derivatives, gradients, second_gradients = self.collect_background_features(model)

            physics_loss = self.compute_physics_loss(quantities, derivatives, gradients, second_gradients)
            bc_loss = self.compute_bc_loss(model)
            loss = physics_loss + self.config.lambda_bc * bc_loss
            losses.append(loss)

            if step % self.config.vtk_save_frequency == 0:
                save_graph_vtk(model.graph_mesh, self.config.output_dir, self.current_epoch, str(batch_idx), step)
                save_background_vtk(model.background_mesh, self.config.output_dir, self.current_epoch, str(batch_idx), step)

        total_loss = torch.mean(torch.stack(losses))
        self.log("train_loss", total_loss, prog_bar=True, batch_size=1)

        if self.current_epoch % self.config.log_case_params_frequency == 0:
            log_case_params([case], self.config.output_dir, self.current_epoch)

        return total_loss

    def validation_step(self, case: Case, batch_idx: int) -> torch.Tensor:
        model = Model(case)

        max_time = self.config.simulation_time
        delta_t = self.config.dt
        n_time_step = int(max_time / delta_t) + 1
        times = torch.linspace(0, max_time, n_time_step, device=self.device)

        current_time = 0.0

        model.update_BC(current_time, self.bc_transform_pressure, self.bc_transform_velocity)
        model.compute_base_features_dt(self.message_passing_conv)
        model.transfer_latent_to_physics(self.transform_conv, self.transform_conv_dt)

        quantities, derivatives, gradients, second_gradients = self.collect_background_features(model)

        ground_truth_ic = torch.zeros_like(quantities)
        physics_loss = self.compute_physics_loss(quantities, derivatives, gradients, second_gradients)
        ic_loss = self.compute_ic_loss(quantities, ground_truth_ic)
        loss = physics_loss + self.config.lambda_ic * ic_loss
        losses = [loss]

        for step in range(n_time_step):
            current_time = times[step]

            model.next_time_step(current_time, delta_t, self.bc_transform_pressure, self.bc_transform_velocity, self.message_passing_conv)

            model.transfer_latent_to_physics(self.transform_conv, self.transform_conv_dt)

            quantities, derivatives, gradients, second_gradients = self.collect_background_features(model)

            physics_loss = self.compute_physics_loss(quantities, derivatives, gradients, second_gradients)
            bc_loss = self.compute_bc_loss(model)
            loss = physics_loss + self.config.lambda_bc * bc_loss
            losses.append(loss)

            if step % self.config.vtk_save_frequency == 0:
                save_graph_vtk(model.graph_mesh, self.config.output_dir, self.current_epoch, f"val_{batch_idx}", step)
                save_background_vtk(model.background_mesh, self.config.output_dir, self.current_epoch, f"val_{batch_idx}", step)

        val_loss = torch.mean(torch.stack(losses))
        self.log("val_loss", val_loss, prog_bar=True, batch_size=1)
        return val_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer