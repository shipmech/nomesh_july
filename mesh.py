# mesh.py
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import knn_graph

import pymesh
import numpy as np

from conditions import BoundaryCondition

def to_torch_int(array, device):
    return torch.tensor(array, dtype=torch.int64, device=device)

def to_torch_float(array, device):
    return torch.tensor(array, dtype=torch.float32, device=device)

class RawMeshGenerator():
    def __call__(self, nodes_positions  : np.array, generate_faces = True):  # nodes_positions: [number of nodes, 2]
        self.nodes = np.array(nodes_positions)

        self.pymesh_mesh= self.generate_pymesh_triangle(self.nodes)
        self.nodes = self.pymesh_mesh.vertices       # np.array[number of nodes, 2]
        #self.edges = self.get_directed_edges()  # np.array[number of edges, 2] # if needed add .T to get [2, number of edges]
        self.faces = None
        if generate_faces:
            self.faces = self.pymesh_mesh.faces

        return self.nodes, self.faces

    def generate_pymesh_triangle(self, vertices):
        tri = pymesh.triangle()
        tri.points = vertices
        tri.split_boundary = False
        tri.verbosity = 0
        tri.run(); # Execute triangle.
        mesh = tri.mesh; # output triangulation.
        return mesh

    def get_directed_edges(self):
        mesh = self.pymesh_mesh
        mesh.enable_connectivity()

        directed_edges_set = set() # set of tuple [number of edges, 2]
        for i in range(len(mesh.vertices)):
            neighbors = mesh.get_vertex_adjacent_vertices(i)
            for neighbor in neighbors:
                indecies_tuple = tuple([i, neighbor].sort())
                directed_edges_set.add(indecies_tuple)

        if len(directed_edges_set) == 0:
            Warning("No edges found in mesh")
            return None

        directed_edges_list = [] # list of lists [number of edges, 2]
        for edge in directed_edges_set:
            directed_edges_list.append(list(edge))
        
        return np.array(directed_edges_list)

class MeshNodeDictData():
    def __init__(self, device, node_pos):
        self.device = device
        
        self.nodes = node_pos   # [number of nodes, 2]

        # Не меняется во время симуляции
        self.dict_BC_type_name_to_BC_type_index = {    # dict of boundary conditions
            'Press' : 0,
            'VelX' : 1,
            'VelY' : 2
        }

        # Не меняется во время симуляции
        self.dict_type_index_to_enabled_BCs = {
            0 : to_torch_int([0,0,0], self.device),
            1 : to_torch_int([1,0,0], self.device),
            2 : to_torch_int([0,1,1], self.device),
        }

        # Граничные условия (в будущем могут меняться во время симуляции)
        self.dict_BC_index_to_BC = {}   # dict of boundary conditions
        self.dict_BC_index_to_BC_type_index = {}   # dict of boundary conditions
        self.dict_BC_index_to_geometry_selection_cond = {} # dict of geometry selection conditions functions

        self.dict_BC_index_to_node_indices_tensor = {} # dict of tensor of node indices
        
        self.dict_node_index_to_enabled_BCs = {}
        for node_index in range(len(self.nodes)):
            self.dict_node_index_to_enabled_BCs[node_index] = to_torch_int([0,0,0], self.device)

        self.dict_node_index_to_type_index = {} # node index to node type index
        self.dict_type_index_to_node_indices_tensor = { # dict of list of node indices
            0 : [],
            1 : [],
            2 : []
        } 

    def add_BC(self, bc : BoundaryCondition, geometry_selection_cond):
        # geometry_selection_cond:
        # lambda nodes, device: torch.isclose(nodes[:, 0], torch.tensor(0.0, device))

        bc_index = len(self.dict_BC_index_to_BC.keys())
        self.dict_BC_index_to_BC[bc_index] = bc
        self.dict_BC_index_to_BC_type_index[bc_index] = self.dict_BC_type_name_to_BC_type_index[self.dict_BC_type_name_to_BC_type_index[bc.type_name]]
        self.dict_BC_index_to_geometry_selection_cond[bc_index] = geometry_selection_cond        

    def apply_bcs(self):
        for bc_index in self.dict_BC_index_to_BC.keys():
            condition_to_select = self.dict_BC_index_to_geometry_selection_cond[bc_index]
            self.dict_BC_index_to_node_indices_tensor[bc_index] = torch.where(condition_to_select(self.nodes, self.device))[0]

            index_in_enabled_bcs = self.dict_BC_index_to_BC_type_index[bc_index]

            for node_index in self.dict_BC_index_to_node_indices_tensor[bc_index]:
                self.dict_node_index_to_enabled_BCs[node_index][index_in_enabled_bcs] = 1

    def determine_node_types(self):
        for node_index in range(len(self.nodes)):
            for bc_type_index, bc_type_tensor in self.dict_type_index_to_enabled_BCs.items():
                if torch.allclose(self.dict_node_index_to_enabled_BCs[node_index], bc_type_tensor):
                    self.dict_node_index_to_type_index[node_index] = bc_type_index
                    self.dict_type_index_to_node_indices_tensor[bc_type_index].append(node_index)
                    break

class GraphData(HeteroData):
    def __init__(self, device, nodes):
        self.device = device

class GraphMesh:
    def __init__(self, device, config, nodes_positions : np.array):
        self.device = device
        self.config = config

        self.initialize_mesh(nodes_positions)
        
    def initialize_mesh(self, nodes_positions : np.array):
        nodes, faces = RawMeshGenerator(nodes_positions)
        self.pos = to_torch_float(nodes, self.device)
        self.edge_index = generate_edges(self.pos, self.config.k_neighbors, self.config.radius)
        self.node_dict_data = MeshNodeDictData(self.device, self.pos)

    def generate_edges(pos: torch.Tensor, k: int, radius: float) -> torch.Tensor:
        # pos: [num_nodes, 2 - num_coordinates]
        # edge_index: [2 is num_vertices, num_edges]
        edge_index = knn_graph(pos, k=k).to(pos.device)
        dist = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)

        mask = dist < radius
        edge_index = edge_index[:, mask].to(pos.device)
        # Assert to check for invalid indices
        if edge_index.numel() > 0:
            assert edge_index.max() < pos.size(0), f"Invalid edge_index: max {edge_index.max()} >= num_nodes {pos.size(0)}"
        return edge_index


class BackgroundMesh():
    def __init__(self, device, nodes, edges=None, faces = None):


        self.nodes = torch.tensor(nodes, dtype=torch.float32, device=device)    # [number of nodes, 2]
        self.edges = None
        if edges is not None:
            self.edges = torch.tensor(edges.T, dtype=torch.int64, device=device)    # [2, number of edges]
        self.faces = None
        if faces is not None:
            self.faces = torch.tensor(faces, dtype=torch.int64, device=device)  # [number of faces, 3]
        pass

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



def generate_edges(pos: torch.Tensor, k: int, radius: float) -> torch.Tensor:
    # pos: [num_nodes, 2 - num_coordinates]
    # edge_index: [2 is num_vertices, num_edges]
    edge_index = knn_graph(pos, k=k).to(pos.device)
    dist = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)

    mask = dist < radius
    edge_index = edge_index[:, mask].to(pos.device)
    # Assert to check for invalid indices
    if edge_index.numel() > 0:
        assert edge_index.max() < pos.size(0), f"Invalid edge_index: max {edge_index.max()} >= num_nodes {pos.size(0)}"
    return edge_index