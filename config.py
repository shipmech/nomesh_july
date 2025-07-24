from dataclasses import dataclass, field
import torch
from typing import Tuple, Dict

@dataclass
class SimulationConfig:
    simulation_time: float = 2.0
    dt: float = 0.01
    vtk_save_frequency: int = 10
    log_case_params_frequency: int = 10
    output_dir: str = "./output"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    domain_width_range: Tuple[float, float] = (0.5, 2.0)
    domain_height_range: Tuple[float, float] = (0.5, 2.0)
    num_nodes_range: Tuple[int, int] = (1000, 2000)
    validation_num_nodes: int = 1500
    distribution_type: str = "uniform"
    radius: float = 0.05
    inlet_velocity_range: Tuple[float, float] = (0.1, 1.0)
    outlet_pressure_range: Tuple[float, float] = (0.0, 100.0)
    number_of_pressure_bc_latent_features: int = 4
    number_of_velocities_bc_latent_features: int = 4
    number_of_base_latent_features: int = 16
    hidden_dim: int = 64
    num_hops: int = 1
    k_neighbors: int = 10
    spline_kernel_size: int = 5
    spline_degree: int = 1
    spline_aggr: str = "mean"
    num_training_cases: int = 5
    num_epochs: int = 100
    learning_rate: float = 0.001
    nx: int = 50
    ny: int = 50
    viscosity: float = 0.01
    density: float = 1.0
    lambda_ic: float = 1.0