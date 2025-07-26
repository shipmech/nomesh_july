# loss.py
import torch
from torch import nn
from torch_geometric.nn import SplineConv
from a_config import SimulationConfig

class PhysicsLoss(nn.Module):
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.viscosity = config.viscosity
        self.density = config.density
        self.dt = config.dt

        # SplineConv for gradient computation (as per improvements)
        in_channels = 1  # Per quantity (u, v, p separately)
        out_channels = 2  # For first derivatives (x, y)
        self.spline_conv_first = SplineConv(in_channels, out_channels, dim=2, kernel_size=config.spline_kernel_size, degree=config.spline_degree, aggr=config.spline_aggr)
        self.spline_conv_second = SplineConv(in_channels, out_channels, dim=2, kernel_size=config.spline_kernel_size, degree=config.spline_degree, aggr=config.spline_aggr)

    def compute_gradients(self, quantities: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, pseudo: torch.Tensor) -> dict[str, torch.Tensor]:
        # quantities: [num_nodes, 3] for u, v, p
        u, v, p = quantities[:, 0:1], quantities[:, 1:2], quantities[:, 2:3]

        # Compute first derivatives
        grad_u = self.spline_conv_first(u, edge_index, pseudo)
        u_x, u_y = grad_u[:, 0], grad_u[:, 1]
        grad_v = self.spline_conv_first(v, edge_index, pseudo)
        v_x, v_y = grad_v[:, 0], grad_v[:, 1]
        grad_p = self.spline_conv_first(p, edge_index, pseudo)
        p_x, p_y = grad_p[:, 0], grad_p[:, 1]

        # Compute second derivatives
        grad_u_x = self.spline_conv_second(grad_u[:, 0:1], edge_index, pseudo)
        u_xx = grad_u_x[:, 0]
        grad_u_y = self.spline_conv_second(grad_u[:, 1:2], edge_index, pseudo)
        u_yy = grad_u_y[:, 1]
        grad_v_x = self.spline_conv_second(grad_v[:, 0:1], edge_index, pseudo)
        v_xx = grad_v_x[:, 0]
        grad_v_y = self.spline_conv_second(grad_v[:, 1:2], edge_index, pseudo)
        v_yy = grad_v_y[:, 1]

        return {
            'u_x': u_x, 'u_y': u_y, 'v_x': v_x, 'v_y': v_y,
            'p_x': p_x, 'p_y': p_y,
            'u_xx': u_xx, 'u_yy': u_yy, 'v_xx': v_xx, 'v_yy': v_yy
        }

    def compute_physics_loss(self, quantities: torch.Tensor, derivatives: torch.Tensor, dt: float, gradients: dict[str, torch.Tensor]) -> torch.Tensor:
        u, v, p = quantities[:, 0], quantities[:, 1], quantities[:, 2]
        u_t, v_t = derivatives[:, 0], derivatives[:, 1]  # Assuming derivatives include u_t, v_t (p_t not used)

        u_x, u_y, v_x, v_y = gradients['u_x'], gradients['u_y'], gradients['v_x'], gradients['v_y']
        p_x, p_y = gradients['p_x'], gradients['p_y']
        u_xx, u_yy, v_xx, v_yy = gradients['u_xx'], gradients['u_yy'], gradients['v_xx'], gradients['v_yy']

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

    def compute_ic_loss(self, predicted_initial: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        return torch.mean((predicted_initial - ground_truth) ** 2)

    def forward(self, quantities: torch.Tensor, prev_quantities: torch.Tensor | None, derivatives: torch.Tensor, dt: float, pos: torch.Tensor, edge_index: torch.Tensor, pseudo: torch.Tensor, is_initial: bool = False, ground_truth_ic: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gradients = self.compute_gradients(quantities, pos, edge_index, pseudo)
        physics_loss = self.compute_physics_loss(quantities, derivatives, dt, gradients)

        ic_loss = torch.tensor(0.0, device=physics_loss.device)
        if is_initial and ground_truth_ic is not None:
            ic_loss = self.compute_ic_loss(quantities[:, :3], ground_truth_ic)  # Assuming first 3 are u,v,p

        total_loss = physics_loss + self.config.lambda_ic * ic_loss

        return total_loss, {'physics_loss': physics_loss, 'ic_loss': ic_loss}