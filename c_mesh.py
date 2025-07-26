# mesh.py
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import knn_graph

import pymesh
import numpy as np

from bb_nn_models import BCTransformingMLP
from d_case import BoundaryCondition
from b_utils import to_torch_int, to_torch_float

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
    def __init__(self, device, node_pos):
        self.device = device
        
        self.nodes = node_pos   # [number of nodes, 2]

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
            0 : to_torch_int([0,0], self.device),
            1 : to_torch_int([1,0], self.device),
            2 : to_torch_int([0,1], self.device),
        }

        # Граничные условия (в будущем могут меняться во время симуляции)
        self.dict_BC_index_to_BC_exact = {}   # dict of boundary conditions
        self.dict_BC_index_to_BCType_index = {}   # dict of boundary conditions
        self.dict_BC_index_to_geometry_selection_cond = {} # dict of geometry selection conditions functions

        self.dict_BC_index_to_node_indices_tensor = {} # dict of tensor of node indices
        self.dict_node_index_to_BC_index = {}

        self.tensor_node_index_to_enabled_BCs = to_torch_int(torch.zeros((len(self.nodes), 2)), self.device)
        self.tensor_node_index_to_type_index = to_torch_int(torch.zeros((len(self.nodes), 1)), self.device) # node index to node type index

        self.dict_type_index_to_node_indices_tensor = { # dict of list of node indices
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
                    self.dict_type_index_to_node_indices_tensor[node_type_index].append(node_index)
                    break


class GraphMesh(nn.Module):
    def __init__(self, device, config, nodes_positions : np.array, bc_list : list):
        self.device = device
        self.config = config

        self.pos : torch.Tensor = None
        self.edge_index : torch.Tensor = None
        self.node_dict_data : MeshNodeDictData = None
        self.graph_hetero_data = None

        self.bc_transform_pressure = None
        self.bc_transform_velocity = None

        self.initialize_mesh(nodes_positions, bc_list)

    def forward(self, t):
        self.update_bc_features(t)
        return self.graph_hetero_data

    def update_bc_features(self, t : float):
        for bc_index, bc_exact in self.node_dict_data.dict_BC_index_to_BC_exact.items():
            value_tensor = bc_exact.get_value(t)
            node_indices_tensor = self.node_dict_data.dict_BC_index_to_node_indices_tensor[bc_index]
            self.graph_hetero_data.bc_features[node_indices_tensor] = value_tensor

    def initialize_mesh(self, nodes_positions : np.array, bc_list : list):
        self.pos = to_torch_float(nodes_positions, self.device)
        self.edge_index = self.generate_edges(self.pos, self.config.k_neighbors, self.config.radius)

        self.node_dict_data = MeshNodeDictData(self.device, self.pos)
        self.determine_BCs(bc_list)
        self.determine_node_types()
        self.graph_hetero_data = self.generate_graph_data()

        self.generate_BC_NNs()

    def generate_edges(self, pos: torch.Tensor, k: int, radius: float) -> torch.Tensor:
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
    
    def determine_BCs(self, bc_list : list): # and determine node types
        # bc_list : list[list[BoundaryCondition, geometry_selection_cond == function]
        for bc in bc_list:
            self.node_dict_data.add_BC(bc[0], bc[1])
        self.node_dict_data.apply_bcs()

    def determine_node_types(self):
        self.node_dict_data.determine_node_types()

    def generate_graph_data(self):
        dict_node_types = self.node_dict_data.dict_type_index_to_type_name
        dict_types_to_nodes = self.node_dict_data.dict_type_index_to_node_indices_tensor

        num_base_f = self.config.number_of_base_latent_features
        dict_node_type_to_num_bc_f = {
            'Free' : 0,
            'Press' : self.config.number_of_pressure_bc_latent_features,
            'NoSlip' : self.config.number_of_velocities_bc_latent_features,
        }
        
        data = HeteroData()

        # set pos for each node type and zero features for each node type
        for node_type_index, node_type_name in dict_node_types.items():
            node_indices = dict_types_to_nodes[node_type_index]

            data[node_type_name].pos = self.pos[node_indices]

            data[node_type_name].base_features = torch.zeros((data[node_type_name].pos.shape[0], num_base_f),
                                                             dtype=torch.float32,
                                                             device=self.device)
            
            data[node_type_name].base_features_dt = torch.zeros((data[node_type_name].pos.shape[0], num_base_f),
                                                                dtype=torch.float32,
                                                                device=self.device)
            
            data[node_type_name].bc_features = torch.zeros((data[node_type_name].pos.shape[0], dict_node_type_to_num_bc_f[node_type_name]),
                                                           dtype=torch.float32,
                                                           device=self.device)

        # set edge indices
        edge_index = self.edge_index
        tensor_node_to_type_index = self.node_dict_data.tensor_node_index_to_type_index

        dict_types_tuple_to_edge_indices_list = {}
        for _, src_type in dict_node_types.items():
            for _, dst_type in dict_node_types.items():
                dict_types_tuple_to_edge_indices_list[(src_type, 'influences', dst_type)] = []
        
        for i_edge in range(edge_index.shape[1]):
            src_type_index = tensor_node_to_type_index[edge_index[0][i_edge]]
            dst_type_index = tensor_node_to_type_index[edge_index[1][i_edge]]

            src_type_name = dict_node_types[src_type_index.item()]
            dst_type_name = dict_node_types[dst_type_index.item()]

            dict_types_tuple_to_edge_indices_list[(src_type_name, 'influences', dst_type_name)].append(i_edge)
        
        for (src_type, relation, dst_type), edges_list in dict_types_tuple_to_edge_indices_list.items():
            indices = to_torch_int(edges_list, self.device)
            data[src_type, relation, dst_type].edge_index = indices
            print(src_type, relation, dst_type, 'shape = ', indices.shape, 'must be (n_edges)')

        return data

    def generate_BC_NNs(self):
        self.bc_transform_pressure = BCTransformingMLP(1, self.config.number_of_pressure_bc_latent_features, self.config.number_bc_nn_hidden_dim)  # pressure BC
        self.bc_transform_velocity = BCTransformingMLP(2, self.config.number_of_velocities_bc_latent_features, self.config.number_bc_nn_hidden_dim)  # velocity BC (u,v)


class BackgroundMesh(GraphMesh):
    def __init__(self, device, config, nodes_positions : np.array, bc_list : list):
        super().__init__(device, config, nodes_positions, bc_list)

    def initialize_mesh(self, nodes_positions : np.array, bc_list : list):
        mesh_generator = TriangleMeshGenerator()
        nodes, edges, faces = mesh_generator(nodes_positions)
        self.pos = to_torch_float(nodes, self.device)
        self.edge_index = to_torch_int(edges, self.device)
        self.faces = to_torch_int(faces, self.device)

        self.node_dict_data = MeshNodeDictData(self.device, self.pos)
        self.determine_BCs(bc_list)
        self.determine_node_types()
        self.graph_hetero_data = self.generate_graph_data()

    def generate_graph_data(self):
        dict_node_types = self.node_dict_data.dict_type_index_to_type_name
        dict_types_to_nodes = self.node_dict_data.dict_type_index_to_node_indices_tensor

        num_phys_f = self.config.num_phys_features                          # u, v, p   (u_t,v_t,p_t - derivatives)
        num_phys_spatial_d = self.config.num_phys_spatial_features_d        # u_x, u_y, v_x, v_y, p_x, p_y
        num_phys_spatial_dd = self.config.num_phys_spatial_features_dd      # u_xx, u_yy, v_xx, v_yy
        
        data = HeteroData()

        # set pos for each node type and zero features for each node type
        for node_type_index, node_type_name in dict_node_types.items():
            node_indices = dict_types_to_nodes[node_type_index]

            data[node_type_name].pos = self.pos[node_indices]

            
            data[node_type_name].phys_features = torch.zeros((data[node_type_name].pos.shape[0], num_phys_f),
                                                             dtype=torch.float32,
                                                             device=self.device)
            
            data[node_type_name].phys_features_dt = torch.zeros((data[node_type_name].pos.shape[0], num_phys_f),
                                                                dtype=torch.float32,
                                                                device=self.device)
            
            data[node_type_name].phys_features_spatial_d = torch.zeros((data[node_type_name].pos.shape[0], num_phys_spatial_d),
                                                                       dtype=torch.float32,
                                                                       device=self.device)
            
            data[node_type_name].phys_features_spatial_dd = torch.zeros((data[node_type_name].pos.shape[0], num_phys_spatial_dd),
                                                                        dtype=torch.float32,
                                                                        device=self.device)

        # set edge indices
        edge_index = self.edge_index
        tensor_node_to_type_index = self.node_dict_data.tensor_node_index_to_type_index
    
        dict_types_tuple_to_edge_indices_list = {}
        for _, src_type in dict_node_types.items():
            for _, dst_type in dict_node_types.items():
                dict_types_tuple_to_edge_indices_list[(src_type, 'influences', dst_type)] = []
        
        for i_edge in range(edge_index.shape[1]):
            src_type_index = tensor_node_to_type_index[edge_index[0][i_edge]]
            dst_type_index = tensor_node_to_type_index[edge_index[1][i_edge]]

            src_type_name = dict_node_types[src_type_index.item()]
            dst_type_name = dict_node_types[dst_type_index.item()]

            dict_types_tuple_to_edge_indices_list[(src_type_name, 'influences', dst_type_name)].append(i_edge)
        
        for (src_type, relation, dst_type), edges_list in dict_types_tuple_to_edge_indices_list.items():
            indices = to_torch_int(edges_list, self.device)
            data[src_type, relation, dst_type].edge_index = indices
            print(src_type, relation, dst_type, 'shape = ', indices.shape, 'must be (n_edges)')

        return data