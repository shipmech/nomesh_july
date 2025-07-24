import os
import torch
import pyvista as pv
import numpy as np
from torch_geometric.data import Data
from scipy.interpolate import griddata
from config import SimulationConfig

def save_vtk(graph_data: Data, mesh_data: Data, output_dir: str, epoch: int, case_id: str, step: int, config: SimulationConfig):
    os.makedirs(os.path.join(output_dir, f"epoch_{epoch}"), exist_ok=True)
    
    # Save graph data
    points = graph_data.pos.cpu().detach().numpy()
    points = np.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=0)  # Add z=0 for 3D visualization
    point_cloud = pv.PolyData(points)
    
    # Save all node features and physical quantities
    total_features = graph_data.x.size(1)
    for i in range(min(total_features, config.number_of_base_latent_features + max(config.number_of_pressure_bc_latent_features, config.number_of_velocities_bc_latent_features))):
        point_cloud.point_data[f'latent_feature_{i}'] = graph_data.x[:, i].cpu().detach().numpy()
    
    for i, name in enumerate(['u', 'v', 'p', 'u_t', 'v_t', 'p_t']):
        idx = config.number_of_base_latent_features + max(config.number_of_pressure_bc_latent_features, config.number_of_velocities_bc_latent_features) + i
        if idx < total_features:
            point_cloud.point_data[name] = graph_data.x[:, idx].cpu().detach().numpy()
    
    point_cloud.point_data['node_type'] = graph_data.node_type.cpu().detach().numpy()
    point_cloud.save(os.path.join(output_dir, f"epoch_{epoch}", f"graph_epoch_{epoch}_case_{case_id}_step_{step}.vtk"))
    
    # Save mesh data on the full 50x50 grid
    nx, ny = config.nx, config.ny  # Use configured 50x50 grid
    width = graph_data.width.item() if torch.is_tensor(graph_data.width) else graph_data.width
    height = graph_data.height.item() if torch.is_tensor(graph_data.height) else graph_data.height
    x = np.linspace(0, width, nx)
    y = np.linspace(0, height, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.zeros_like(X)
    grid = pv.StructuredGrid(X, Y, Z)

    # Interpolate mesh_data.x to the 50x50 grid
    mesh_points = mesh_data.pos.cpu().detach().numpy() if hasattr(mesh_data, 'pos') else graph_data.pos.cpu().detach().numpy()
    for i, name in enumerate(['u', 'v', 'p', 'u_t', 'v_t', 'p_t']):
        values = mesh_data.x[:, i].cpu().detach().numpy()
        interpolated_values = griddata(mesh_points, values, (X, Y), method='linear', fill_value=0.0)
        grid.point_data[name] = interpolated_values.ravel('F')

    grid.save(os.path.join(output_dir, f"epoch_{epoch}", f"bg_mesh_epoch_{epoch}_case_{case_id}_step_{step}.vtk"))

def log_case_params(cases: list[Data], output_dir: str, epoch: int):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"case_params_epoch_{epoch}.txt")
    
    with open(log_file, 'w') as f:
        for case in cases:
            boundary_data = case.boundary_data.cpu().detach()
            node_type = case.node_type.cpu().detach()
            pos = case.pos.cpu().detach()
            width = case.width.item() if torch.is_tensor(case.width) else case.width
            height = case.height.item() if torch.is_tensor(case.height) else case.height
            inlet_mask = (node_type == 2) & (pos[:, 0] < 0.01)
            outlet_mask = (node_type == 1) & (pos[:, 0] > width - 0.01)
            top_bottom_mask = (node_type == 2) & ((pos[:, 1] < 0.01) | (pos[:, 1] > height - 0.01))
            
            inlet_u = boundary_data[inlet_mask, 0].mean().item() if inlet_mask.any() else 0.0
            inlet_v = boundary_data[inlet_mask, 1].mean().item() if inlet_mask.any() else 0.0
            outlet_p = boundary_data[outlet_mask, 0].mean().item() if outlet_mask.any() else 0.0
            
            f.write(f"Case ID: {case.case_id}\n")
            f.write(f"Width: {width:.4f}\n")
            f.write(f"Height: {height:.4f}\n")
            f.write(f"Number of nodes: {case.num_nodes}\n")
            f.write(f"Distribution type: {case.distribution_type}\n")
            f.write(f"Inlet velocity (u, v): ({inlet_u:.4f}, {inlet_v:.4f})\n")
            f.write(f"Outlet pressure: {outlet_p:.4f}\n")
            f.write(f"Number of inlet nodes: {inlet_mask.sum().item()}\n")
            f.write(f"Number of outlet nodes: {outlet_mask.sum().item()}\n")
            f.write(f"Number of top/bottom nodes: {top_bottom_mask.sum().item()}\n")
            f.write("\n")