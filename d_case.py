from abc import ABC, abstractmethod
import torch

import numpy as np
from c_mesh import GraphMesh, BackgroundMesh
from c_mesh import PressureBC, VelocityBC
from b_utils import to_torch_int, to_torch_float

class Geometry(ABC):
    @abstractmethod
    def get_bounds(self) -> tuple[float, float, float, float]:
        pass

class Box(Geometry):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def get_bounds(self) -> tuple[float, float, float, float]:
        return 0.0, self.width, 0.0, self.height

class Sphere(Geometry):
    def __init__(self, radius: float, center: tuple[float, float]):
        self.radius = radius
        self.center = center

    def get_bounds(self) -> tuple[float, float, float, float]:
        # Bounding box for sphere
        cx, cy = self.center
        return cx - self.radius, cx + self.radius, cy - self.radius, cy + self.radius

class Case:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.device)

        self.geometry: Geometry | None = None

        self.graph_mesh: GraphMesh | None = None
        self.background_mesh: BackgroundMesh | None = None

        self.boundary_conditions: list = []
        self.times: torch.Tensor | None = None

        self.width: float | None = None
        self.height: float | None = None
        self.inlet_vel: float | None = None
        self.outlet_press: float | None = None

        self._initialize()

    def _initialize(self):
        self.width = np.random.uniform(*self.config.domain_width_range)
        self.height = np.random.uniform(*self.config.domain_height_range)
        self.geometry = Box(self.width, self.height)

        num_steps = int(self.config.simulation_time / self.config.dt) + 1
        self.times = torch.linspace(0, self.config.simulation_time, num_steps, device=self.device)
        num_steps_for_bc = 2
        self.times_for_bc = torch.linspace(0, 1e8, num_steps_for_bc, device=self.device)

        # Calculate nx, ny based on num_nodes
        if hasattr(self.config, 'validation_num_nodes') and self.config.validation_num_nodes:
            num_nodes = self.config.validation_num_nodes
        else:
            num_nodes = np.random.randint(*self.config.num_nodes_range)
        aspect = self.width / self.height
        ny = int(np.sqrt(num_nodes / aspect))
        nx = int(num_nodes / ny)

        self.num_nodes = nx * ny

        x = torch.linspace(0, self.geometry.width, nx, device=self.device)
        y = torch.linspace(0, self.geometry.height, ny, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        vertices = torch.stack([xx.flatten(), yy.flatten()], dim=1).cpu().numpy()
        
        bc_geometry_selection_cond = {
            'inlet': lambda nodes, device: torch.isclose(nodes[:, 0], to_torch_float([0.0], device)),
            'outlet': lambda nodes, device: torch.isclose(nodes[:, 0], to_torch_float([self.geometry.width], device)),
            'top': lambda nodes, device: torch.isclose(nodes[:, 1],  to_torch_float([self.geometry.height], device)),
            'bottom': lambda nodes, device: torch.isclose(nodes[:, 1], to_torch_float([0.0], device))
        }
        
        # Set boundary conditions
        vel_values = torch.zeros((num_steps_for_bc, 2), device=self.device)
        top_bc = [VelocityBC(self.times_for_bc, vel_values[:, 0], self.times_for_bc, vel_values[:, 1]), bc_geometry_selection_cond['top']]
        self.boundary_conditions.append(top_bc)
        bottom_bc = [VelocityBC(self.times_for_bc, vel_values[:, 0], self.times_for_bc, vel_values[:, 1]), bc_geometry_selection_cond['bottom']]
        self.boundary_conditions.append(bottom_bc)

        self.inlet_vel = np.random.uniform(*self.config.inlet_velocity_range)
        vel_values = torch.zeros((num_steps_for_bc, 2), device=self.device)
        vel_values[:, 0] = self.inlet_vel  # u
        vel_values[:, 1] = 0.0  # v
        inlet_bc = [VelocityBC(self.times_for_bc, vel_values[:, 0], self.times_for_bc, vel_values[:, 1]), bc_geometry_selection_cond['inlet']]
        self.boundary_conditions.append(inlet_bc)

        self.outlet_press = np.random.uniform(*self.config.outlet_pressure_range)
        press_values = torch.zeros((num_steps_for_bc, 1), device=self.device)
        press_values[:, 0] = self.outlet_press  # p
        outlet_bc = [PressureBC(self.times_for_bc, press_values), bc_geometry_selection_cond['outlet']]
        self.boundary_conditions.append(outlet_bc)

        # Create meshes
        nodes_positions = vertices
        self.graph_mesh = GraphMesh(self.device, self.config, nodes_positions, self.boundary_conditions)
        self.background_mesh = BackgroundMesh(self.device, self.config, nodes_positions, self.boundary_conditions)