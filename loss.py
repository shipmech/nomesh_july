import torch
from config import SimulationConfig

class PhysicsLoss:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.dt = config.dt
        self.viscosity = config.viscosity

    def compute_loss(self, quantities: torch.Tensor, prev_quantities: torch.Tensor, derivatives: torch.Tensor, dt: float, target_points: torch.Tensor = None):
        """
        Compute physics-based loss over the entire domain.
        quantities: [time_steps, num_nodes, features] where features include [u, v, p, u_t, v_t, p_t]
        prev_quantities: [time_steps, num_nodes, features] with shifted time steps
        derivatives: [time_steps, num_nodes, derivatives] (e.g., spatial gradients)
        target_points: Optional tensor of target points for domain coverage
        """
        time_steps, num_nodes, _ = quantities.size()
        
        # Extract components
        u = quantities[:, :, 0]  # u velocity
        v = quantities[:, :, 1]  # v velocity
        p = quantities[:, :, 2]  # pressure
        u_t = quantities[:, :, 3]  # u time derivative
        v_t = quantities[:, :, 4]  # v time derivative
        p_t = quantities[:, :, 5]  # p time derivative

        prev_u = prev_quantities[:, :, 0]
        prev_v = prev_quantities[:, :, 1]
        prev_p = prev_quantities[:, :, 2]

        # Assume derivatives include du_dx, du_dy, dv_dx, dv_dy, dp_dx, dp_dy
        du_dx = derivatives[:, :, 0]
        du_dy = derivatives[:, :, 1]
        dv_dx = derivatives[:, :, 2]
        dv_dy = derivatives[:, :, 3]
        dp_dx = derivatives[:, :, 4]
        dp_dy = derivatives[:, :, 5]

        # Continuity equation: du_dx + dv_dy = 0
        continuity_residual = du_dx + dv_dy
        continuity_loss = torch.mean(continuity_residual ** 2)

        # Momentum equations (Navier-Stokes)
        momentum_x_residual = u_t + u * du_dx + v * du_dy + (1 / self.config.density) * dp_dx - self.viscosity * (du_dx ** 2 + du_dy ** 2)
        momentum_y_residual = v_t + u * dv_dx + v * dv_dy + (1 / self.config.density) * dp_dy - self.viscosity * (dv_dx ** 2 + dv_dy ** 2)
        momentum_x_loss = torch.mean(momentum_x_residual ** 2)
        momentum_y_loss = torch.mean(momentum_y_residual ** 2)

        # Time derivative consistency (optional, based on numerical integration)
        time_deriv_loss = torch.mean((u - prev_u - u_t * dt) ** 2) + torch.mean((v - prev_v - v_t * dt) ** 2) + torch.mean((p - prev_p - p_t * dt) ** 2)

        total_physics_loss = continuity_loss + momentum_x_loss + momentum_y_loss + time_deriv_loss
        return total_physics_loss, [continuity_loss, momentum_x_loss, momentum_y_loss, time_deriv_loss]