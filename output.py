# output.py
import torch
from torch_geometric.data import Data
import pyvista as pv
from scipy.interpolate import griddata
import numpy as np
import os
from config import SimulationConfig
from typing import List
from case import Case

def save_vtk(graph_data: Data, mesh_data: Data, output_dir: str, epoch: int, case_id: str, step: int, config: SimulationConfig):
    os.makedirs(output_dir, exist_ok=True)

    # Pad pos to 3D with z=0 for graph data
    pos_cpu = graph_data.pos.detach().cpu().numpy()
    pos_3d = np.hstack([pos_cpu, np.zeros((pos_cpu.shape[0], 1), dtype=pos_cpu.dtype)])

    # Save graph data (nodes with features)
    graph_filename = os.path.join(output_dir, f"graph_epoch{epoch}_case{case_id}_step{step}.vtk")
    points = pv.PolyData(pos_3d)
    features = graph_data.x.detach().cpu().numpy()
    points['features'] = features  # 14D features
    node_type = graph_data.node_type.detach().cpu().numpy()
    points['node_type'] = node_type
    points.save(graph_filename)

    # Save mesh data (interpolated to background mesh)
    mesh_filename = os.path.join(output_dir, f"mesh_epoch{epoch}_case{case_id}_step{step}.vtk")
    # Assuming background mesh is 50x50 grid
    nx, ny = config.nx, config.ny
    width = graph_data.width.item()
    height = graph_data.height.item()
    x = np.linspace(0, width, nx)
    y = np.linspace(0, height, ny)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    interpolated_quantities = griddata(mesh_data.pos.detach().cpu().numpy(), mesh_data.x.detach().cpu().numpy(), grid_points, method='nearest')

    # Create a flat 3D grid
    grid = pv.RectilinearGrid(x, y, [0])
    grid.point_data['quantities'] = interpolated_quantities  # Directly set the (2500, 6) array

    grid.save(mesh_filename)

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