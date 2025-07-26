from dataclasses import dataclass, field
import torch
from typing import Tuple, Dict

@dataclass
class SimulationConfig:
    simulation_time: float = 2.0
    dt: float = 0.01

    vtk_save_frequency: int = 1
    log_case_params_frequency: int = 1
    output_dir: str = "./output"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    #device: str = "cpu"

    domain_width_range: Tuple[float, float] = (0.5, 2.0)
    domain_height_range: Tuple[float, float] = (0.5, 2.0)

    num_nodes_range: Tuple[int, int] = (1000, 2000)
    validation_num_nodes: int = 1500
    #distribution_type: str = "uniform"
    
    inlet_velocity_range: Tuple[float, float] = (-1.0, 1.0)
    outlet_pressure_range: Tuple[float, float] = (-100.0, 100.0)

    number_bc_nn_hidden_dim: int = 64

    k_neighbors: int = 10
    radius: float = 0.05

    number_of_base_latent_features: int = 16
    number_of_pressure_bc_latent_features: int = 2
    number_of_velocities_bc_latent_features: int = 4

    message_passing_hidden_dim: int = 64

    num_phys_features: int = 3                  # u, v, p   (u_t,v_t,p_t - derivatives)
    num_phys_spatial_features_d: int = 6        # u_x, u_y, v_x, v_y, p_x, p_y
    num_phys_spatial_features_dd: int = 4       # u_xx, u_yy, v_xx, v_yy

    knn_radius_transfer: float = 0.1
    kernel_size_transfer: int = 5

    num_training_cases: int = 1
    num_epochs: int = 100
    learning_rate: float = 0.01

    viscosity: float = 0.01
    density: float = 1.0
    lambda_ic: float = 1.0
    lambda_bc: float = 1.0      #Weight for BC loss