# output.py
import torch
from torch_geometric.data import Data, HeteroData
import pyvista as pv
from scipy.interpolate import griddata
import numpy as np
import os
from a_config import SimulationConfig
from typing import List
from d_case import Case
from c_mesh import GraphMesh, BackgroundMesh

def save_graph_vtk(graph_mesh: GraphMesh, output_dir: str, epoch: int, case_id: str, step: int):
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"graph_epoch{epoch}_case{case_id}_step{step}.vtk")

    data = graph_mesh.graph_hetero_data

    pos = graph_mesh.pos.detach().cpu().numpy()
    pos_3d = np.c_[pos, np.zeros(pos.shape[0])]

    N = pos.shape[0]

    points = pv.PolyData(pos_3d)

    # Add edges as lines
    edge_index = graph_mesh.edge_index.detach().cpu().numpy().T  # [num_edges, 2]
    if edge_index.shape[0] > 0:
        lines = np.hstack([np.full((edge_index.shape[0], 1), 2, dtype=int), edge_index]).ravel()
        points.lines = lines

    node_type = graph_mesh.node_dict_data.tensor_node_index_to_type_index.detach().cpu().numpy().squeeze()
    points['node_type'] = node_type

    dict_node_types = graph_mesh.node_dict_data.dict_type_index_to_type_name

    num_base_f = data['Free'].base_features.shape[1] if 'Free' in data.node_types else graph_mesh.config.number_of_base_latent_features

    base_features = np.zeros((N, num_base_f), dtype=np.float32)
    base_features_dt = np.zeros((N, num_base_f), dtype=np.float32)
    phys_features = np.zeros((N, 3), dtype=np.float32)
    phys_features_dt = np.zeros((N, 3), dtype=np.float32)

    for node_type_index, node_type_name in dict_node_types.items():
        type_nodes = np.array(graph_mesh.node_dict_data.dict_type_index_to_node_indices_tensor[node_type_index])
        if len(type_nodes) == 0:
            continue
        base_features[type_nodes] = data[node_type_name].base_features.detach().cpu().numpy()
        base_features_dt[type_nodes] = data[node_type_name].base_features_dt.detach().cpu().numpy()
        phys_features[type_nodes] = data[node_type_name].phys_features.detach().cpu().numpy()
        phys_features_dt[type_nodes] = data[node_type_name].phys_features_dt.detach().cpu().numpy()

    points['base_features'] = base_features
    points['base_features_dt'] = base_features_dt
    points['phys_features'] = phys_features
    points['phys_features_dt'] = phys_features_dt

    # Add individual components for phys_features
    points['u'] = phys_features[:, 0]
    points['v'] = phys_features[:, 1]
    points['p'] = phys_features[:, 2]

    # Add individual components for phys_features_dt
    points['u_t'] = phys_features_dt[:, 0]
    points['v_t'] = phys_features_dt[:, 1]
    points['p_t'] = phys_features_dt[:, 2]

    if 'Press' in data.node_types:
        press_dim = data['Press'].bc_features.shape[1]
        press_bc = np.zeros((N, press_dim), dtype=np.float32)
        press_nodes = np.array(graph_mesh.node_dict_data.dict_type_index_to_node_indices_tensor[1])
        press_bc[press_nodes] = data['Press'].bc_features.detach().cpu().numpy()
        points['press_bc_features'] = press_bc

    if 'NoSlip' in data.node_types:
        noslip_dim = data['NoSlip'].bc_features.shape[1]
        noslip_bc = np.zeros((N, noslip_dim), dtype=np.float32)
        noslip_nodes = np.array(graph_mesh.node_dict_data.dict_type_index_to_node_indices_tensor[2])
        noslip_bc[noslip_nodes] = data['NoSlip'].bc_features.detach().cpu().numpy()
        points['noslip_bc_features'] = noslip_bc

    # Add BC index
    bc_index_array = np.full(N, -1, dtype=np.int32)
    for bc_idx, node_tensor in graph_mesh.node_dict_data.dict_BC_index_to_node_indices_tensor.items():
        nodes = node_tensor.detach().cpu().numpy()
        bc_index_array[nodes] = bc_idx
    points['bc_index'] = bc_index_array

    points.save(filename)

def save_background_vtk(background_mesh: BackgroundMesh, output_dir: str, epoch: int, case_id: str, step: int):
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"background_epoch{epoch}_case{case_id}_step{step}.vtk")

    data = background_mesh.graph_hetero_data

    pos = background_mesh.pos.detach().cpu().numpy()
    pos_3d = np.c_[pos, np.zeros(pos.shape[0])]

    N = pos.shape[0]

    faces = background_mesh.faces.detach().cpu().numpy()
    num_faces = faces.shape[0]
    cell_types = np.full(num_faces, pv.CellType.TRIANGLE, dtype=np.uint8)
    cells = np.hstack([np.full((num_faces, 1), 3), faces]).ravel()

    grid = pv.UnstructuredGrid(cells, cell_types, pos_3d)

    node_type = background_mesh.node_dict_data.tensor_node_index_to_type_index.detach().cpu().numpy().squeeze()
    grid.point_data['node_type'] = node_type

    dict_node_types = background_mesh.node_dict_data.dict_type_index_to_type_name

    phys_features = np.zeros((N, 3), dtype=np.float32)
    phys_features_dt = np.zeros((N, 3), dtype=np.float32)
    phys_spatial_d = np.zeros((N, 6), dtype=np.float32)
    phys_spatial_dd = np.zeros((N, 4), dtype=np.float32)

    for node_type_index, node_type_name in dict_node_types.items():
        type_nodes = np.array(background_mesh.node_dict_data.dict_type_index_to_node_indices_tensor[node_type_index])
        if len(type_nodes) == 0:
            continue
        phys_features[type_nodes] = data[node_type_name].phys_features.detach().cpu().numpy()
        phys_features_dt[type_nodes] = data[node_type_name].phys_features_dt.detach().cpu().numpy()
        phys_spatial_d[type_nodes] = data[node_type_name].phys_features_spatial_d.detach().cpu().numpy()
        phys_spatial_dd[type_nodes] = data[node_type_name].phys_features_spatial_dd.detach().cpu().numpy()

    grid.point_data['phys_features'] = phys_features
    grid.point_data['phys_features_dt'] = phys_features_dt
    grid.point_data['phys_features_spatial_d'] = phys_spatial_d
    grid.point_data['phys_features_spatial_dd'] = phys_spatial_dd

    # Add individual components for phys_features
    grid.point_data['u'] = phys_features[:, 0]
    grid.point_data['v'] = phys_features[:, 1]
    grid.point_data['p'] = phys_features[:, 2]

    # Add individual components for phys_features_dt
    grid.point_data['u_t'] = phys_features_dt[:, 0]
    grid.point_data['v_t'] = phys_features_dt[:, 1]
    grid.point_data['p_t'] = phys_features_dt[:, 2]

    # Add individual components for phys_features_spatial_d
    grid.point_data['u_x'] = phys_spatial_d[:, 0]
    grid.point_data['u_y'] = phys_spatial_d[:, 1]
    grid.point_data['v_x'] = phys_spatial_d[:, 2]
    grid.point_data['v_y'] = phys_spatial_d[:, 3]
    grid.point_data['p_x'] = phys_spatial_d[:, 4]
    grid.point_data['p_y'] = phys_spatial_d[:, 5]

    # Add individual components for phys_features_spatial_dd
    grid.point_data['u_xx'] = phys_spatial_dd[:, 0]
    grid.point_data['u_yy'] = phys_spatial_dd[:, 1]
    grid.point_data['v_xx'] = phys_spatial_dd[:, 2]
    grid.point_data['v_yy'] = phys_spatial_dd[:, 3]

    # Add BC index
    bc_index_array = np.full(N, -1, dtype=np.int32)
    for bc_idx, node_tensor in background_mesh.node_dict_data.dict_BC_index_to_node_indices_tensor.items():
        nodes = node_tensor.detach().cpu().numpy()
        bc_index_array[nodes] = bc_idx
    grid.point_data['bc_index'] = bc_index_array

    grid.save(filename)

def log_case_params(cases: List[Case], output_dir: str, epoch: int):
    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, f"case_params_epoch{epoch}.txt")
    with open(log_filename, 'w') as f:
        for i, case in enumerate(cases):
            f.write(f"Case {i}:\n")
            f.write(f"Width: {case.width}\n")
            f.write(f"Height: {case.height}\n")
            f.write(f"Inlet Velocity: {case.inlet_vel}\n")
            f.write(f"Outlet Pressure: {case.outlet_press}\n")
            f.write(f"Number of Nodes: {case.graph_data.num_nodes.item()}\n")
            f.write("\n")