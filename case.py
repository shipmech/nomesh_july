# case.py (updated with numpy import and empty faces as 2D array)
import torch
from torch_geometric.data import Data
import pymesh
import random
import numpy as np  # Added import
from geometry import Box, Sphere, Geometry
from mesh import Mesh, update_edges
from conditions import BoundaryCondition, InitialCondition, PressureBC, VelocityBC
from utils import TimeHistoryData

class Case:
    def __init__(self, config):
        self.config = config
        self.geometry: Geometry | None = None
        self.mesh: Mesh | None = None
        self.boundary_conditions: list[BoundaryCondition] = []
        self.initial_condition: InitialCondition = InitialCondition()
        self.time_history: TimeHistoryData | None = None
        self.graph_data: Data | None = None

        self._initialize()

    def _initialize(self):
        # Generate geometry (default: random Box for training)
        width = random.uniform(*self.config.domain_width_range)
        height = random.uniform(*self.config.domain_height_range)
        self.geometry = Box(width, height)

        # Generate time history (example: constant for simplicity; extend for time-dependent)
        num_steps = int(self.config.simulation_time / self.config.dt) + 1
        times = torch.linspace(0, self.config.simulation_time, num_steps)
        # Example values (extend based on BC type)
        values = torch.zeros((num_steps, 3))  # e.g., [u, v, p]
        self.time_history = TimeHistoryData(times, values)

        # Generate mesh using PyMesh (uniform grid points for 2D)
        x = torch.linspace(0, self.geometry.width, self.config.nx)
        y = torch.linspace(0, self.geometry.height, self.config.ny)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        vertices = torch.stack([xx.flatten(), yy.flatten()], dim=1).numpy()
        faces = np.array([], dtype=np.int32).reshape(0, 3)  # Empty 2D array for faces

        pymesh_mesh = pymesh.form_mesh(vertices=vertices, faces=faces)

        # Boundary node sets
        nodes = torch.tensor(vertices, dtype=torch.float32)
        boundary_node_sets = {
            'inlet': set(torch.where(nodes[:, 0] == 0)[0].tolist()),
            'outlet': set(torch.where(nodes[:, 0] == self.geometry.width)[0].tolist()),
            'top': set(torch.where(nodes[:, 1] == self.geometry.height)[0].tolist()),
            'bottom': set(torch.where(nodes[:, 1] == 0)[0].tolist()),
        }
        self.mesh = Mesh(pymesh_mesh, boundary_node_sets)

        # Add boundary conditions (random for training)
        inlet_vel = random.uniform(*self.config.inlet_velocity_range)
        outlet_press = random.uniform(*self.config.outlet_pressure_range)
        vel_values = torch.full_like(values, inlet_vel)  # Constant for now
        press_values = torch.full_like(values, outlet_press)
        self.boundary_conditions.append(VelocityBC(TimeHistoryData(times, vel_values[:, :2])))  # u, v
        self.boundary_conditions.append(PressureBC(TimeHistoryData(times, press_values[:, 2:3])))  # p

        # Generate graph
        self.graph_data = self.generate_graph()

    def generate_graph(self) -> Data:
        pos = self.mesh.get_nodes()
        edge_index = update_edges(pos, self.config.k_neighbors, self.config.radius)

        # Node types (0: free, 1: pressure, 2: velocity)
        node_type = torch.zeros(pos.shape[0], dtype=torch.long)
        node_type[list(self.mesh.boundary_node_sets['outlet'])] = 1  # Pressure
        node_type[list(self.mesh.boundary_node_sets['inlet'])] = 2  # Velocity
        node_type[list(self.mesh.boundary_node_sets['top'].union(self.mesh.boundary_node_sets['bottom']))] = 2  # Walls as velocity=0

        # Features (latent + BC + physical; aligned with version 3)
        base_dim = self.config.number_of_base_latent_features
        bc_dim = max(self.config.number_of_pressure_bc_latent_features, self.config.number_of_velocities_bc_latent_features)
        phys_dim = 6  # u,v,p,u_t,v_t,p_t
        x = torch.zeros((pos.shape[0], base_dim + 2 * bc_dim + phys_dim), dtype=torch.float32)

        # Set initial physical quantities
        initial_state = self.initial_condition.get_initial_state().repeat(pos.shape[0], 1)
        x[:, -phys_dim:-3] = initial_state  # u,v,p initial
        # u_t,v_t,p_t initial = 0

        return Data(x=x, pos=pos, edge_index=edge_index, node_type=node_type)

    def update(self, t: float):
        # Update boundary conditions in graph_data
        for bc in self.boundary_conditions:
            value = bc.get_value(t)
            if isinstance(bc, VelocityBC):
                indices = list(self.mesh.boundary_node_sets['inlet'])
                self.graph_data.x[indices, -6:-4] = value  # Update u,v in physical quantities
            elif isinstance(bc, PressureBC):
                indices = list(self.mesh.boundary_node_sets['outlet'])
                self.graph_data.x[indices, -3] = value  # Update p

    def get_mesh(self) -> Mesh:
        return self.mesh

    def get_graph_data(self) -> Data:
        return self.graph_data

    def add_obstacle(self, geometry: Geometry, trajectory: TimeHistoryData):
        self.mesh.exclude_obstacles([geometry])
        # Update graph after exclusion (recompute edge_index)
        self.graph_data = self.generate_graph()
        # Trajectory for moving (handled in update for future)