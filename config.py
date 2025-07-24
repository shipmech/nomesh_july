import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SimulationConfig:
    # Simulation parameters
    simulation_time: float = 2.0
    dt: float = 0.02  # Increased for faster testing
    vtk_save_frequency: int = 10  # Save VTK every 10th time step
    log_case_params_frequency: int = 10
    output_dir: str = "./output"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Domain parameters
    domain_width_range: Tuple[float, float] = (0.5, 2.0)
    domain_height_range: Tuple[float, float] = (0.5, 2.0)
    num_nodes_range: Tuple[int, int] = (100, 300)  # Reduced for faster testing
    validation_width: float = 1.0
    validation_height: float = 1.0
    validation_num_nodes: int = 200  # Reduced for faster testing
    distribution_type: str = "uniform"
    radius: float = 0.1

    # Boundary conditions
    inlet_velocity_range: Tuple[float, float] = (0.1, 1.0)
    outlet_pressure_range: Tuple[float, float] = (0.0, 100.0)
    number_of_pressure_bc_latent_features: int = 4
    number_of_velocities_bc_latent_features: int = 4

    # Model parameters
    number_of_base_latent_features: int = 4  # Reduced for faster testing
    hidden_dim: int = 16  # Reduced for faster testing
    num_hops: int = 2  # Reduced for faster testing
    k_neighbors: int = 8
    spline_kernel_size: int = 4
    spline_degree: int = 3
    spline_aggr: str = "mean"

    # Training parameters
    num_training_cases: int = 2  # Reduced for testing
    num_epochs: int = 5  # Reduced for testing
    learning_rate: float = 0.001
    nx: int = 50
    ny: int = 50
    viscosity: float = 0.01
    density: float = 1.0