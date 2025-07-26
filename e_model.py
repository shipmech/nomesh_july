# case.py
import torch
import torch.nn as nn
from torch_geometric.nn import radius
from torch_geometric.nn import HeteroConv

from d_case import Case
from bb_nn_models import TransformConv, MessagePassingMLPConv

class Model():
    def __init__(self, case : Case):
        self.config = case.config
        self.device = case.device

        self.graph_mesh = case.graph_mesh
        self.background_mesh = case.background_mesh

        self.max_time = self.config.simulation_time
        self.dt = self.config.dt
        self.n_time_step = int(self.max_time / self.time_step) + 1
        self.times = torch.linspace(0, self.max_time, self.n_time_step, device=self.device)

        self.current_time = 0.0

        self.shared_NN = {} # must be initialized in forward, must contain
                            # 'transform_conv', 'transform_conv_dt', 'reverse_transform_conv', 'message_passing_conv'

    def set_shared_NN(self, shared_NN):
        self.shared_NN = shared_NN

    def generate_shared_NNs(self):
        # Construct SplineConv for transfer (bipartite, source to target)
        num_base_f = self.config.number_of_base_latent_features

        # phys + spatial_d + spatial_dd
        physical_channels = self.config.num_phys_features + \
                            self.config.num_phys_spatial_features_d + \
                            self.config.num_phys_spatial_features_dd
        
        physical_dt_channels = self.config.num_phys_features  # Assuming phys_features_dt has same dim as phys_features

        dim = 2  # 2 for (x,y)
        kernel_size = self.config.kernel_size_transfer

        transform_conv = TransformConv((num_base_f, 0), physical_channels, dim=dim, kernel_size=[kernel_size,kernel_size]).to(self.device)
        transform_conv_dt = TransformConv((num_base_f, 0), physical_dt_channels, dim=dim, kernel_size=[kernel_size,kernel_size]).to(self.device) # Separate conv for dt
        reverse_transform_conv = TransformConv((physical_channels, 0), num_base_f, dim=dim, kernel_size=[kernel_size, kernel_size]).to(self.device)

        # define_message_passing_conv
        data = self.graph_mesh.graph_hetero_data
        node_types, edge_types = data.metadata()
        
        base_dim = self.config.number_of_base_latent_features
        bc_dims = {
            'Free': 0,
            'Press': self.config.number_of_pressure_bc_latent_features,
            'NoSlip': self.config.number_of_velocities_bc_latent_features,
        }
        
        conv_dict = {}
        for edge_type in edge_types:
            src, _, dst = edge_type
            src_channels = base_dim + bc_dims[src]
            conv_dict[edge_type] = MessagePassingMLPConv(
                src_channels=src_channels,
                out_channels=base_dim,
                hidden_dim=self.config.hidden_dim,
                aggr='add'  # or 'mean' if preferred
            )
        
        message_passing_conv = HeteroConv(conv_dict, aggr='sum')

        bc_transform_pressure, bc_transform_velocity = self.graph_mesh.generate_shared_NNs()

        dict_shared_NN = {
            'bc_transform_pressure': bc_transform_pressure,
            'bc_transform_velocity': bc_transform_velocity,
            'transform_conv': transform_conv,
            'transform_conv_dt': transform_conv_dt,
            'reverse_transform_conv': reverse_transform_conv,
            'message_passing_conv': message_passing_conv,
        }

        return dict_shared_NN
        
    def transfer_latent_to_physics(self):
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

        x = (source_base, None)
        x_dt = (source_base_dt, None)

        out = self.shared_NN['transform_conv'](x=x, edge_index=assignment, edge_attr=edge_attr, size=(num_source, num_target))
        out_dt = self.shared_NN['transform_conv_dt'](x=x_dt, edge_index=assignment, edge_attr=edge_attr, size=(num_source, num_target))

        # Split and assign to target per type
        data_target = self.background_mesh.graph_hetero_data
        for t in node_types:
            s = target_offsets[t]
            data_target[t].phys_features = out[s, 0:self.config.num_phys_features]
            data_target[t].phys_features_spatial_d = out[s, self.config.num_phys_features:self.config.num_phys_features + self.config.num_phys_spatial_features_d]
            data_target[t].phys_features_spatial_dd = out[s, self.config.num_phys_features + self.config.num_phys_spatial_features_d:]
            data_target[t].phys_features_dt = out_dt[s]

    def transfer_physics_to_latent(self):
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

        x = (source_phys, None)

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

    def compute_base_features_dt(self):
        data = self.graph_mesh.graph_hetero_data
        node_types = data.node_types  # ['Free', 'Press', 'NoSlip']
        
        # Prepare x_dict: cat(base_features, bc_features) per node type
        # Note: bc_features assumed to be latent
        x_dict = {}
        for t in node_types:
            base = data[t].base_features
            bc = data[t].bc_features
            x_dict[t] = torch.cat([base, bc], dim=-1) if bc.size(1) > 0 else base
        
        # Run message passing
        dt_dict = self.message_passing_conv(x_dict, data.edge_index_dict)
        
        # Assign dt
        for t in node_types:
            data[t].base_features_dt = dt_dict[t]

    def next_time_step(self):
        # update BC
        # Update base features by integraition RK4
        # Update base features dt by compute_base_features_dt at new time step

        # Assumes current time step is managed externally; here we advance base_features using RK4
        # Need current t; assume self.current_time is set (add self.current_time = 0.0 in _initialize if needed)
        dt = self.dt
        data = self.graph_mesh.graph_hetero_data
        node_types = data.node_types
        
        # Store original base_features
        original_base = {t: data[t].base_features.clone() for t in node_types}
        
        # RK4 steps; since BC(t) changes, update BC at intermediate times
        # k1 = f(t, y)
        self.update_BC(self.current_time)
        self.compute_base_features_dt()
        k1 = {t: data[t].base_features_dt.clone() for t in node_types}
        
        # k2 = f(t + dt/2, y + dt/2 * k1)
        for t in node_types:
            data[t].base_features = original_base[t] + (dt / 2) * k1[t]
        #self.update_BC(self.current_time + dt / 2)
        self.compute_base_features_dt()
        k2 = {t: data[t].base_features_dt.clone() for t in node_types}
        
        # k3 = f(t + dt/2, y + dt/2 * k2)
        for t in node_types:
            data[t].base_features = original_base[t] + (dt / 2) * k2[t]
        #self.update_BC(self.current_time + dt / 2)
        self.compute_base_features_dt()
        k3 = {t: data[t].base_features_dt.clone() for t in node_types}
        
        # k4 = f(t + dt, y + dt * k3)
        for t in node_types:
            data[t].base_features = original_base[t] + dt * k3[t]
        #self.update_BC(self.current_time + dt)
        self.compute_base_features_dt()
        k4 = {t: data[t].base_features_dt.clone() for t in node_types}
        
        # Update: y += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for t in node_types:
            data[t].base_features = original_base[t] + (dt / 6) * (k1[t] + 2 * k2[t] + 2 * k3[t] + k4[t])
        
        # Advance time and recompute dt at new time step
        self.current_time += dt
        self.update_BC(self.current_time)
        self.compute_base_features_dt()