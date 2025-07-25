# case.py
import torch
from torch_geometric.data import HeteroData
import pymesh
import numpy as np
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
        self.graph_data: HeteroData | None = None

        self._initialize()

    def _initialize(self):
        self.width = np.random.uniform(*self.config.domain_width_range)
        self.height = np.random.uniform(*self.config.domain_height_range)
        self.geometry = Box(self.width, self.height)

        num_steps = int(self.config.simulation_time / self.config.dt) + 1
        times = torch.linspace(0, self.config.simulation_time, num_steps, device=self.device)
        values = torch.zeros((num_steps, 3), device=self.device)
        self.time_history = TimeHistoryData(times, values)

        # Calculate nx, ny based on num_nodes
        if hasattr(self.config, 'validation_num_nodes') and self.config.validation_num_nodes:
            num_nodes = self.config.validation_num_nodes
        else:
            num_nodes = np.random.randint(*self.config.num_nodes_range)
        aspect = self.width / self.height
        ny = int(np.sqrt(num_nodes / aspect))
        nx = int(num_nodes / ny)

        x = torch.linspace(0, self.geometry.width, nx, device=self.device)
        y = torch.linspace(0, self.geometry.height, ny, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        vertices = torch.stack([xx.flatten(), yy.flatten()], dim=1).cpu().numpy()
        faces = np.array([], dtype=np.int32).reshape(0, 3)
        pymesh_mesh = pymesh.form_mesh(vertices=vertices, faces=faces)

        nodes = torch.tensor(vertices, dtype=torch.float32, device=self.device)
        node_sets = {
            'free' : set(torch.arange(nodes.shape[0], device=self.device).cpu().tolist()),
            'inlet': set(torch.where(torch.isclose(nodes[:, 0], torch.tensor(0.0, device=self.device)))[0].cpu().tolist()),
            'outlet': set(torch.where(torch.isclose(nodes[:, 0], torch.tensor(self.geometry.width, device=self.device)))[0].cpu().tolist()),
            'top': set(torch.where(torch.isclose(nodes[:, 1], torch.tensor(self.geometry.height, device=self.device)))[0].cpu().tolist()),
            'bottom': set(torch.where(torch.isclose(nodes[:, 1], torch.tensor(0.0, device=self.device)))[0].cpu().tolist()),
        }

        node_sets['inlet'] = node_sets['inlet'] - node_sets['top'] - node_sets['bottom']
        node_sets['outlet'] = node_sets['outlet'] - node_sets['top'] - node_sets['bottom']
        node_sets['wall'] = node_sets['top'] | node_sets['bottom']

        node_sets['free'] = node_sets['free'] - node_sets['inlet'] - node_sets['outlet'] - node_sets['wall']

        self.mesh = Mesh(pymesh_mesh, node_sets, self.device)

        self.inlet_vel = np.random.uniform(*self.config.inlet_velocity_range)
        self.outlet_press = np.random.uniform(*self.config.outlet_pressure_range)
        vel_values = torch.zeros((num_steps, 2), device=self.device)
        vel_values[:, 0] = self.inlet_vel  # u
        vel_values[:, 1] = 0.0  # v
        press_values = torch.zeros((num_steps, 1), device=self.device)
        press_values[:, 0] = self.outlet_press  # p
        self.boundary_conditions.append(VelocityBC(times, vel_values))
        self.boundary_conditions.append(PressureBC(times, press_values))

        self.graph_data = self.generate_graph()

    def generate_graph(self) -> HeteroData:
        node_types = ['free', 'inlet', 'outlet', 'wall']

        pos = self.mesh.get_nodes()
        node_sets = self.mesh.node_sets
        edge_index = update_edges(pos, self.config.k_neighbors, self.config.radius)

        node_index_to_node_type_dict = {}
        for node_type in node_types:
            node_index_to_node_type_dict.update({i: node_type for i in node_sets[node_type]})
        
        data = HeteroData()

        # set pos for each node type
        for node_type in node_types:
            data[node_type].pos = pos[list(node_sets[node_type])]

        # set zero x-features for each node type
        def set_x(node_type, x_dim):
            data[node_type].x = torch.zeros((data[node_type].pos.shape[0], x_dim), dtype=torch.float32, device=self.device)

        set_x('free', self.config.number_of_free_latent_features)
        set_x('inlet', self.config.number_of_base_latent_features + self.config.number_of_velocities_bc_latent_features)
        set_x('outlet', self.config.number_of_base_latent_features + self.config.number_of_pressure_bc_latent_features)
        set_x('wall', self.config.number_of_wall_latent_features)
        
        # set edge indices
        types_to_edge_indices_dict_of_list = {}
        for src_type in node_types:
            for dst_type in node_types:
                types_to_edge_indices_dict_of_list[(src_type, 'influences', dst_type)] = []
        
        for i_edge in range(edge_index.shape[1]):
            src_type = node_index_to_node_type_dict[edge_index[0][i_edge]]
            dst_type = node_index_to_node_type_dict[edge_index[1][i_edge]]
            types_to_edge_indices_dict_of_list[(src_type, 'influences', dst_type)].append(i_edge)
        
        for (src_type, relation, dst_type) in types_to_edge_indices_dict_of_list:
            indices = torch.tensor(types_to_edge_indices_dict_of_list[(src_type, relation, dst_type)], dtype=torch.long, device=self.device)
            data[src_type, relation, dst_type].edge_index = indices
            print(src_type, relation, dst_type, 'shape = ', indices.shape, 'must be (2, n_edges)')
        
        #phys_dim = 6  # u,v,p,u_t,v_t,p_t

        initial_state = self.initial_condition.get_initial_state().repeat(pos.shape[0], 1)
        #x[:, -phys_dim:-3] = initial_state  # u,v,p initial
        # u_t,v_t,p_t initial = 0

        graph_data = data
        graph_data.width = torch.tensor(self.width, device=self.device)
        graph_data.height = torch.tensor(self.height, device=self.device)
        graph_data.inlet_velocity = torch.tensor(self.inlet_vel, device=self.device)
        graph_data.outlet_pressure = torch.tensor(self.outlet_press, device=self.device)
        graph_data.num_nodes = torch.tensor(pos.shape[0], device=self.device)
        return graph_data

    def update(self, t: float):
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

    def get_graph_data(self) -> HeteroData:
        return self.graph_data

    def add_obstacle(self, geometry: Geometry, trajectory: TimeHistoryData):
        #self.mesh.exclude_obstacles([geometry])
        # Update graph after exclusion (recompute edge_index)
        #self.graph_data = self.generate_graph()
        # Trajectory for moving (handled in update for future)

        # Placeholder for exclude obstacles
        pass