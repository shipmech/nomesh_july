# case.py
import torch
import torch.nn as nn
from torch_geometric.nn import radius

import numpy as np
from geometry import Box, Geometry
from mesh import GraphMesh, BackgroundMesh
from conditions import PressureBC, VelocityBC
from utils import to_torch_int, to_torch_float
from nn_models import TransformConv

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

class Model(nn.Module):
    def __init__(self, case : Case):
        self.config = case.config
        self.device = case.device

        self.graph_mesh: nn.Module = case.graph_mesh
        self.background_mesh: nn.Module = case.background_mesh

        self.transform_conv: TransformConv = None
        self.transform_conv_dt: TransformConv = None
        self.reverse_transform_conv: TransformConv = None
        self.transform_dict_data = {}
        self.reverse_transform_dict_data = {}

        self.max_time = self.config.simulation_time
        self.dt = self.config.dt
        self.n_time_step = int(self.max_time / self.time_step) + 1
        self.times = torch.linspace(0, self.max_time, self.n_time_step, device=self.device)
        
        self._initialize()

    def _initialize(self):
        self.define_transform_conv_latent_to_physics()
        self.define_transform_conv_physics_to_latent()

    def define_transform_conv_latent_to_physics(self):
        data_source = self.graph_mesh.graph_hetero_data
        data_target = self.background_mesh.graph_hetero_data
        node_types = ['Free', 'Press', 'NoSlip']  # From your MeshNodeDictData

        # Step 1: Collect source pos and base_features (concat across types)
        source_pos = torch.cat([data_source[t].pos for t in node_types], dim=0)
        source_base = torch.cat([data_source[t].base_features for t in node_types], dim=0)
        source_base_dt = torch.cat([data_source[t].base_features_dt for t in node_types], dim=0)
        num_source = source_pos.size(0)

        # Collect target pos (concat across types)
        target_pos = torch.cat([data_target[t].pos for t in node_types], dim=0)
        num_target = target_pos.size(0)

        # Compute offsets for splitting back to types
        target_offsets = {}
        offset = 0
        for t in node_types:
            n = data_target[t].pos.size(0)
            target_offsets[t] = slice(offset, offset + n)
            offset += n

        # Step 2: Build bipartite edges (source -> target) using radius
        r = self.config.knn_radius_transfer  # Your radius parameter
        max_num_neighbors = 32  # Max per target; prevents explosion in dense areas
        assignment = radius(source_pos, target_pos, r, max_num_neighbors=max_num_neighbors)
        # assignment: [2, num_assignments], row 0: source idx, row 1: target idx

        # Optional: Check for isolated targets (zero neighbors) and handle (e.g., log or assign default)
        row_counts = torch.bincount(assignment[1], minlength=num_target)
        isolated = torch.where(row_counts == 0)[0]
        if len(isolated) > 0:
            print(f"Warning: {len(isolated)} target nodes have no source neighbors within radius {r}.")
            # Could reselect: e.g., connect to global nearest, or skip/exclude them

        # Step 3: Compute relative positions and normalize to [0,1] by radius
        rel_pos = target_pos[assignment[1]] - source_pos[assignment[0]]
        r = self.config.knn_radius_transfer
        edge_attr = (rel_pos / (2 * r)) + 0.5  # Shifts [-r, r] to [0,1]; clamp if needed
        edge_attr = torch.clamp(edge_attr, 0.0, 1.0)  # Ensure bounds

        # Step 4: Construct SplineConv for transfer (bipartite, source to target)
        num_base_f = self.config.number_of_base_latent_features

        # phys + spatial_d + spatial_dd
        out_channels = self.config.num_phys_features + \
                       self.config.num_phys_spatial_features_d + \
                       self.config.num_phys_spatial_features_dd 

        dim = source_pos.size(1)  # 2 for (x,y)
        kernel_size = self.config.kernel_size_transfer

        self.transform_conv = TransformConv((num_base_f, 0), out_channels, dim=dim, kernel_size=[kernel_size,kernel_size]).to(self.device)

        # Separate conv for dt
        out_channels_dt = self.config.num_phys_features  # Assuming phys_features_dt has same dim as phys_features
        self.transform_conv_dt = TransformConv((num_base_f, 0), out_channels_dt, dim=dim, kernel_size=[kernel_size,kernel_size]).to(self.device)

        x = (source_base, None)
        x_dt = (source_base_dt, None)

        self.transform_dict_data = {
            'node_types': node_types,
            'x': x,
            'x_dt': x_dt,
            'edge_index': assignment,
            'edge_attr': edge_attr,
            'size': (num_source, num_target),
            'target_offsets': target_offsets
        }

    def define_transform_conv_physics_to_latent(self):
        data_source = self.background_mesh.graph_hetero_data  # Now background is source (physics)
        data_target = self.graph_mesh.graph_hetero_data  # Graph is target (latents)
        node_types = ['Free', 'Press', 'NoSlip']  # Same node types

        # Step 1: Collect source pos and physics features (concat across types)
        source_pos = torch.cat([data_source[t].pos for t in node_types], dim=0)
        source_phys = torch.cat([
            torch.cat([
                data_source[t].phys_features,
                data_source[t].phys_features_spatial_d,
                data_source[t].phys_features_spatial_dd
            ], dim=1) for t in node_types
        ], dim=0)
        num_source = source_pos.size(0)

        # Collect target pos (concat across types)
        target_pos = torch.cat([data_target[t].pos for t in node_types], dim=0)
        num_target = target_pos.size(0)

        # Compute offsets for splitting back to types
        target_offsets = {}
        offset = 0
        for t in node_types:
            n = data_target[t].pos.size(0)
            target_offsets[t] = slice(offset, offset + n)
            offset += n

        # Step 2: Build bipartite edges (source -> target) using radius
        r = self.config.knn_radius_transfer  # Same radius
        max_num_neighbors = 32  # Same max
        assignment = radius(source_pos, target_pos, r, max_num_neighbors=max_num_neighbors)
        # assignment: [2, num_assignments], row 0: source idx (background), row 1: target idx (graph)

        # Optional: Check for isolated targets
        row_counts = torch.bincount(assignment[1], minlength=num_target)
        isolated = torch.where(row_counts == 0)[0]
        if len(isolated) > 0:
            print(f"Warning: {len(isolated)} target nodes have no source neighbors within radius {r}.")

        # Step 3: Compute relative positions and normalize to [0,1] by radius
        rel_pos = target_pos[assignment[1]] - source_pos[assignment[0]]
        edge_attr = (rel_pos / (2 * r)) + 0.5
        edge_attr = torch.clamp(edge_attr, 0.0, 1.0)

        # Step 4: Construct SplineConv for reverse transfer
        num_base_f = self.config.number_of_base_latent_features

        # Input channels: phys + spatial_d + spatial_dd
        in_channels = self.config.num_phys_features + \
                      self.config.num_phys_spatial_features_d + \
                      self.config.num_phys_spatial_features_dd

        dim = source_pos.size(1)  # 2 for (x,y)
        kernel_size = self.config.kernel_size_transfer  # Same kernel size

        self.reverse_transform_conv = TransformConv((in_channels, 0), num_base_f, dim=dim, kernel_size=[kernel_size, kernel_size]).to(self.device)

        x = (source_phys, None)

        self.reverse_transform_dict_data = {
            'node_types': node_types,
            'x': x,
            'edge_index': assignment,
            'edge_attr': edge_attr,
            'size': (num_source, num_target),
            'target_offsets': target_offsets
        }

    def define_message_passing_conv(self): # TODO: implement
        pass

    def transfer_latent_to_physics(self):
        # Forward pass
        node_types = self.transform_dict_data['node_types']
        x = self.transform_dict_data['x']
        x_dt = self.transform_dict_data['x_dt']
        assignment = self.transform_dict_data['edge_index']
        edge_attr = self.transform_dict_data['edge_attr']
        num_source = self.transform_dict_data['size'][0]
        num_target = self.transform_dict_data['size'][1]
        target_offsets = self.transform_dict_data['target_offsets']

        out = self.transform_conv(x=x, edge_index=assignment, edge_attr=edge_attr, size=(num_source, num_target))
        out_dt = self.transform_conv_dt(x=x_dt, edge_index=assignment, edge_attr=edge_attr, size=(num_source, num_target))

        # Split and assign to target per type
        data_target = self.background_mesh.graph_hetero_data
        for t in node_types:
            s = target_offsets[t]
            data_target[t].phys_features = out[s, 0:self.config.num_phys_features]
            data_target[t].phys_features_spatial_d = out[s, self.config.num_phys_features:self.config.num_phys_features + self.config.num_phys_spatial_features_d]
            data_target[t].phys_features_spatial_dd = out[s, self.config.num_phys_features + self.config.num_phys_spatial_features_d:]
            data_target[t].phys_features_dt = out_dt[s]

    def transfer_physics_to_latent(self):
        # Forward pass for reverse
        node_types = self.reverse_transform_dict_data['node_types']
        x = self.reverse_transform_dict_data['x']
        assignment = self.reverse_transform_dict_data['edge_index']
        edge_attr = self.reverse_transform_dict_data['edge_attr']
        num_source = self.reverse_transform_dict_data['size'][0]
        num_target = self.reverse_transform_dict_data['size'][1]
        target_offsets = self.reverse_transform_dict_data['target_offsets']

        out = self.reverse_transform_conv(x=x, edge_index=assignment, edge_attr=edge_attr, size=(num_source, num_target))

        # Assign to target (graph_mesh) base_features
        data_target = self.graph_mesh.graph_hetero_data
        for t in node_types:
            s = target_offsets[t]
            data_target[t].base_features = out[s]

    def update_initial_conditions(self):
        self.transfer_physics_to_latent(self)
        self.transfer_latent_to_physics(self)

    def update_BC(self, t):
        self.graph_mesh.update_bc_features(t)

    def compute_base_features_dt(self):  # TODO: implement
        # temp_graph_hetero_data = self.graph_mesh.graph_hetero_data
        # ...
        pass

    def next_time_step(self):  # TODO: implement
        # update BC
        # Update base features by integraition RK4
        # Update base features dt by compute_base_features_dt at new time step
        pass