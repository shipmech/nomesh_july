# case.py
import torch
from torch_geometric.data import Data
import pymesh
import random
import numpy as np
import math
from geometry import Box, Sphere, Geometry
from mesh import Mesh, update_edges
from conditions import BoundaryCondition, InitialCondition, PressureBC, VelocityBC
from utils import TimeHistoryData

class Case:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.device)
        self.geometry: Geometry | None = None
        self.mesh: Mesh | None = None
        self.boundary_conditions: list[BoundaryCondition] = []
        self.initial_condition: InitialCondition = InitialCondition(device=self.device)
        self.time_history: TimeHistoryData | None = None
        self.graph_data: Data | None = None

        self._initialize()

    def _initialize(self):
        self.width = random.uniform(*self.config.domain_width_range)
        self.height = random.uniform(*self.config.domain_height_range)
        self.geometry = Box(self.width, self.height)

        num_steps = int(self.config.simulation_time / self.config.dt) + 1
        times = torch.linspace(0, self.config.simulation_time, num_steps, device=self.device)
        values = torch.zeros((num_steps, 3), device=self.device)
        self.time_history = TimeHistoryData(times, values)

        # Calculate nx, ny based on num_nodes
        if hasattr(self.config, 'validation_num_nodes') and self.config.validation_num_nodes:
            num_nodes = self.config.validation_num_nodes
        else:
            num_nodes = random.randint(*self.config.num_nodes_range)
        aspect = self.width / self.height
        ny = int(math.sqrt(num_nodes / aspect))
        nx = int(num_nodes / ny)

        x = torch.linspace(0, self.geometry.width, nx, device=self.device)
        y = torch.linspace(0, self.geometry.height, ny, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        vertices = torch.stack([xx.flatten(), yy.flatten()], dim=1).cpu().numpy()
        faces = np.array([], dtype=np.int32).reshape(0, 3)
        pymesh_mesh = pymesh.form_mesh(vertices=vertices, faces=faces)

        nodes = torch.tensor(vertices, dtype=torch.float32, device=self.device)
        boundary_node_sets = {
            'inlet': set(torch.where(torch.isclose(nodes[:, 0], torch.tensor(0.0, device=self.device)))[0].cpu().tolist()),
            'outlet': set(torch.where(torch.isclose(nodes[:, 0], torch.tensor(self.geometry.width, device=self.device)))[0].cpu().tolist()),
            'top': set(torch.where(torch.isclose(nodes[:, 1], torch.tensor(self.geometry.height, device=self.device)))[0].cpu().tolist()),
            'bottom': set(torch.where(torch.isclose(nodes[:, 1], torch.tensor(0.0, device=self.device)))[0].cpu().tolist()),
        }
        self.mesh = Mesh(pymesh_mesh, boundary_node_sets, self.device)

        self.inlet_vel = random.uniform(*self.config.inlet_velocity_range)
        self.outlet_press = random.uniform(*self.config.outlet_pressure_range)
        vel_values = torch.zeros_like(values)
        vel_values[:, 0] = self.inlet_vel  # u
        vel_values[:, 1] = 0.0  # v
        press_values = torch.zeros_like(values)
        press_values[:, 0] = self.outlet_press  # p
        self.boundary_conditions.append(VelocityBC(TimeHistoryData(times, vel_values[:, :2])))
        self.boundary_conditions.append(PressureBC(TimeHistoryData(times, press_values[:, :1])))

        self.graph_data = self.generate_graph()

    def generate_graph(self) -> Data:
        pos = self.mesh