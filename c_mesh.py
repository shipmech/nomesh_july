from abc import ABC, abstractmethod
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import knn_graph

import pymesh
import numpy as np

from b_utils import TimeHistoryData

class BoundaryCondition(ABC):
    def __init__(self):
        pass

    def get_value(self, t: float):
        pass

    def get_derivative(self, t: float):
        pass

class PressureBC(BoundaryCondition):
    def __init__(self, times: torch.Tensor, values: torch.Tensor):
        self.time_history = TimeHistoryData(times, values)
        self.type_name = 'Press'

    def get_value(self, t: float):
        return self.time_history.get_value(t)
    
    def get_derivative(self, t: float):
        return self.time_history.get_derivative(t)

class VelocityBC(BoundaryCondition):
    def __init__(self,
                 times_u: torch.Tensor, values_u: torch.Tensor,
                 times_v: torch.Tensor, values_v: torch.Tensor):
        self.time_history_u = TimeHistoryData(times_u, values_u)
        self.time_history_v = TimeHistoryData(times_v, values_v)
        self.type_name = 'Vel'

    def get_value(self, t: float):
        u = self.time_history_u.get_value(t)
        v = self.time_history_v.get_value(t)
        return torch.tensor([u, v], dtype=u.dtype, device=u.device)
    
    def get_derivative(self, t: float):
        dudt = self.time_history_u.get_derivative(t)
        dvdt = self.time_history_v.get_derivative(t)
        return torch.tensor([dudt, dvdt], dtype=dudt.dtype, device=dudt.device)

class TriangleMeshGenerator():
    def __call__(self, nodes_positions  : np.array):  # nodes_positions: [number of nodes, 2]
        self.nodes = np.array(nodes_positions)

        self.pymesh_mesh= self.generate_pymesh_triangle(self.nodes)
        self.nodes = self.pymesh_mesh.vertices       # np.array[number of nodes, 2]
        self.edges = self.get_directed_edges().T  # np.array[2, number of edges]
        self.faces = self.pymesh_mesh.faces

        return self.nodes, self.edges, self.faces

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
                list_of_indices = [i, neighbor]
                list_of_indices.sort()
                indecies_tuple = tuple(list_of_indices)
                directed_edges_set.add(indecies_tuple)

        if len(directed_edges_set) == 0:
            Warning("No edges found in mesh")
            return None

        directed_edges_list = [] # list of lists [number of edges, 2]
        for edge in directed_edges_set:
            directed_edges_list.append(list(edge))
        
        return np.array(directed_edges_list)


class MeshNodeDictData():
    def __init__(self, device, node_pos, edge_index):
        self.device = device
        
        self.nodes = node_pos   # [number of nodes, 2]
        self.pos = node_pos
        self.edge_index = edge_index  # [2, number of edges]

        # Не меняется во время симуляции
        self.dict_BCType_name_to_BC_type_index = {    # dict of boundary conditions
            'Press' : 0,
            'Vel' : 1,
        }

        # Не меняется во время симуляции
        self.dict_type_index_to_type_name = {       # node type
            0 : 'Free',
            1 : 'Press',
            2 : 'NoSlip'
        }
        
        self.dict_type_index_to_enabled_BCs = {
            0 : torch.tensor([0,0], dtype=torch.int64, device=self.device),
            1 : torch.tensor([1,0], dtype=torch.int64, device=self.device),
            2 : torch.tensor([0,1], dtype=torch.int64, device=self.device),
        }

        # Граничные условия (в будущем могут меняться во время симуляции)
        self.dict_BC_index_to_BC_exact = {}   # dict of boundary conditions
        self.dict_BC_index_to_BCType_index = {}   # dict of boundary conditions
        self.dict_BC_index_to_geometry_selection_cond = {} # dict of geometry selection conditions functions

        self.dict_BC_index_to_node_indices_tensor = {} # dict of tensor of node indices
        # self.dict_node_index_to_BC_index = {} 

        self.tensor_node_index_to_enabled_BCs = torch.zeros((len(self.nodes), 2), dtype=torch.int64, device=self.device)
        self.tensor_node_index_to_type_index = torch.zeros((len(self.nodes), 1), dtype=torch.int64, device=self.device) # node index to node type index

        self.dict_type_index_to_node_indices_list = { # dict of list of node indices
            0 : [],
            1 : [],
            2 : []
        }

        self.dict_type_index_to_bc_indices_list = { # dict of list of node indices
            0 : [],
            1 : [],
            2 : []
        }

    def add_BC(self, bc : BoundaryCondition, geometry_selection_cond):
        # geometry_selection_cond:
        # lambda nodes, device: torch.isclose(nodes[:, 0], torch.tensor(0.0, device))

        bc_index = len(self.dict_BC_index_to_BC_exact.keys())
        self.dict_BC_index_to_BC_exact[bc_index] = bc
        self.dict_BC_index_to_BCType_index[bc_index] = self.dict_BCType_name_to_BC_type_index[bc.type_name]
        self.dict_BC_index_to_geometry_selection_cond[bc_index] = geometry_selection_cond        

    def apply_bcs(self):
        for bc_index in self.dict_BC_index_to_BC_exact.keys():
            condition_to_select = self.dict_BC_index_to_geometry_selection_cond[bc_index]
            # select nodes by condition from all nodes
            self.dict_BC_index_to_node_indices_tensor[bc_index] = torch.where(condition_to_select(self.nodes, self.device))[0]

        for bc_index in self.dict_BC_index_to_BC_exact.keys():
            # remove nodes from other bcs which has been applied previously
            for bc_index_other in self.dict_BC_index_to_BC_exact.keys():
                if bc_index_other < bc_index:
                   mask = torch.isin(self.dict_BC_index_to_node_indices_tensor[bc_index], self.dict_BC_index_to_node_indices_tensor[bc_index_other], invert=True)
                   self.dict_BC_index_to_node_indices_tensor[bc_index] = torch.masked_select(self.dict_BC_index_to_node_indices_tensor[bc_index], mask)
            # now apply bc in that order: walls, inlet, outlet

        for bc_index, node_indices in self.dict_BC_index_to_node_indices_tensor.items():
            index_in_enabled_bcs = self.dict_BC_index_to_BCType_index[bc_index]
            for node_index in node_indices:
                self.tensor_node_index_to_enabled_BCs[node_index][index_in_enabled_bcs] = 1

    def determine_node_types(self):
        for node_index in range(len(self.nodes)):
            for node_type_index, bc_type_tensor in self.dict_type_index_to_enabled_BCs.items():
                if torch.allclose(self.tensor_node_index_to_enabled_BCs[node_index], bc_type_tensor):
                    self.tensor_node_index_to_type_index[node_index] = node_type_index
                    self.dict_type_index_to_node_indices_list[node_type_index].append(node_index)
                    
                    for bc_index, node_indices in self.dict_BC_index_to_node_indices_tensor.items():
                        if torch.isin(node_index, node_indices, assume_unique=True).max():
                            self.dict_type_index_to_bc_indices_list[node_type_index].append(bc_index)

                    break

class GraphData():
    def __init__(self, config, node_dict_data, pos, edge_index, faces):
        self.config = config
        self.node_dict_data = node_dict_data
        self.pos = pos
        self.edge_index = edge_index
        self.faces = faces

def initialize_graph_hetero_data(config, type : str, nodes_positions : np.array, bc_list : list):
    # type : str = 'bacground' or 'background_mesh'
    # bc_list : list[list[BoundaryCondition, geometry_selection_cond == function]

    device = config.device
    pos = None
    edge_index = None
    faces = None
    if type == 'latent_graph':
        pos = torch.tensor(nodes_positions, dtype=torch.float32, device=device)
        edge_index = generate_edges(pos, config.k_neighbors, config.radius)
    elif type == 'background_mesh':
        mesh_generator = TriangleMeshGenerator()
        nodes, edges, faces = mesh_generator(nodes_positions)
        pos = torch.tensor(nodes, dtype=torch.float32, device=device)
        edge_index = torch.tensor(edges, dtype=torch.long, device=device)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        faces = torch.tensor(faces, dtype=torch.long, device=device)

    node_dict_data = MeshNodeDictData(device, pos, edge_index)

    for bc in bc_list:
        node_dict_data.add_BC(bc[0], bc[1])
    node_dict_data.apply_bcs()

    node_dict_data.determine_node_types()

    data = HeteroData()

    graph_hetero_data = generate_graph_hetero_data(config, device, data, node_dict_data, type=type)

    graph_data = GraphData(config, node_dict_data, pos, edge_index, faces)

    graph_data.config = config
    graph_data.node_dict_data = node_dict_data
    graph_data.pos = pos
    graph_data.edge_index = edge_index
    graph_data.faces = faces
    
    return graph_hetero_data, graph_data

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

def generate_graph_hetero_data(config, device, data, node_dict_data, type : str): # type : str = 'latent_graph' or 'background_mesh'
    
    dict_node_types = node_dict_data.dict_type_index_to_type_name
    dict_types_to_nodes = node_dict_data.dict_type_index_to_node_indices_list
    tensor_node_to_type_index = node_dict_data.tensor_node_index_to_type_index
    pos, edge_index = node_dict_data.pos, node_dict_data.edge_index

    data = set_features_for_graph_hetero_data(config, type, data, dict_node_types, dict_types_to_nodes, pos)
    data = set_edges_for_graph_hetero_data(device, data, tensor_node_to_type_index, dict_node_types, dict_types_to_nodes, edge_index)
    return data

def set_features_for_graph_hetero_data(config, type : str, data, dict_node_types, dict_types_to_nodes, pos):
    device = config.device
    
    if type == 'latent_graph':
        num_base_f = config.number_of_base_latent_features
        dict_node_type_to_num_bc_f = {
            'Free' : 0,
            'Press' : config.number_of_pressure_bc_latent_features,
            'NoSlip' : config.number_of_velocities_bc_latent_features,
        }

        # set pos for each node type and zero features for each node type
        for node_type_index, node_type_name in dict_node_types.items():
            node_indices = dict_types_to_nodes[node_type_index]

            data[node_type_name].pos = pos[node_indices]

            data[node_type_name].base_features = torch.zeros((data[node_type_name].pos.shape[0], num_base_f),
                                                                dtype=torch.float32,
                                                                device=device)
            
            data[node_type_name].base_features_dt = torch.zeros((data[node_type_name].pos.shape[0], num_base_f),
                                                                dtype=torch.float32,
                                                                device=device)
            
            data[node_type_name].bc_features = torch.zeros((data[node_type_name].pos.shape[0], dict_node_type_to_num_bc_f[node_type_name]),
                                                            dtype=torch.float32,
                                                            device=device)

        return data

    elif type == 'background_mesh':
        device = config.device
        num_phys_f = config.num_phys_features                          # u, v, p   (u_t,v_t,p_t - derivatives)
        num_phys_spatial_d = config.num_phys_spatial_features_d        # u_x, u_y, v_x, v_y, p_x, p_y
        num_phys_spatial_dd = config.num_phys_spatial_features_dd      # u_xx, u_yy, v_xx, v_yy
        
        dict_node_type_to_num_bc_f = {
            'Free' : 0,
            'Press' : 1,
            'NoSlip' : 2,
        }

        # set pos for each node type and zero features for each node type
        for node_type_index, node_type_name in dict_node_types.items():
            node_indices = dict_types_to_nodes[node_type_index]

            data[node_type_name].pos = pos[node_indices]

            
            data[node_type_name].phys_features = torch.zeros((data[node_type_name].pos.shape[0], num_phys_f),
                                                                dtype=torch.float32,
                                                                device=device)
            
            data[node_type_name].phys_features_dt = torch.zeros((data[node_type_name].pos.shape[0], num_phys_f),
                                                                dtype=torch.float32,
                                                                device=device)
            
            data[node_type_name].phys_features_spatial_d = torch.zeros((data[node_type_name].pos.shape[0], num_phys_spatial_d),
                                                                        dtype=torch.float32,
                                                                        device=device)
            
            data[node_type_name].phys_features_spatial_dd = torch.zeros((data[node_type_name].pos.shape[0], num_phys_spatial_dd),
                                                                        dtype=torch.float32,
                                                                        device=device)

            data[node_type_name].bc_features = torch.zeros((data[node_type_name].pos.shape[0], dict_node_type_to_num_bc_f[node_type_name]),
                                                            dtype=torch.float32,
                                                            device=device)
            
        return data

def set_edges_for_graph_hetero_data(device, data, tensor_node_to_type_index, dict_node_types, dict_types_to_nodes, edge_index):
    # set edge indices
    dict_types_tuple_to_edge_indices_list = {}
    for _, src_type in dict_node_types.items():
        for _, dst_type in dict_node_types.items():
            dict_types_tuple_to_edge_indices_list[(src_type, 'influences', dst_type)] = []
    
    for i_edge in range(edge_index.shape[1]):
        src_node_index = edge_index[0][i_edge]
        dst_node_index = edge_index[1][i_edge]

        src_type_index = tensor_node_to_type_index[src_node_index].item()
        dst_type_index = tensor_node_to_type_index[dst_node_index].item()

        src_type_name = dict_node_types[src_type_index]
        dst_type_name = dict_node_types[dst_type_index]

        src_node_indices = torch.tensor(dict_types_to_nodes[src_type_index], dtype=torch.float32, device=device)
        dst_node_indices = torch.tensor(dict_types_to_nodes[dst_type_index], dtype=torch.float32, device=device)

        src_node_new_index = (src_node_indices == src_node_index).nonzero(as_tuple=True)[0]
        dst_node_new_index = (dst_node_indices == dst_node_index).nonzero(as_tuple=True)[0]

        dict_types_tuple_to_edge_indices_list[(src_type_name, 'influences', dst_type_name)].append([src_node_new_index, dst_node_new_index])
    
    for (src_type, relation, dst_type), node_indecies_list_of_list in dict_types_tuple_to_edge_indices_list.items():
        edges = torch.tensor(node_indecies_list_of_list, dtype=torch.long, device=device)
        
        data[src_type, relation, dst_type].edge_index = edges.t().contiguous()
        #print(src_type, relation, dst_type, 'shape = ', indices.shape, 'must be (2, n_edges)')

    return data
