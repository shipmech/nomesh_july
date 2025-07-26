# case.py
import torch
from torch_geometric.nn import radius

from d_case import Case

class Model():
    def __init__(self, case : Case):
        self.config = case.config
        self.device = case.device

        self.graph_mesh = case.graph_mesh
        self.background_mesh = case.background_mesh
        
    def transfer_latent_to_physics(self, transform_conv, transform_conv_dt):
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

        out = transform_conv(x=x, edge_index=assignment, edge_attr=edge_attr, size=(num_source, num_target))
        out_dt = transform_conv_dt(x=x_dt, edge_index=assignment, edge_attr=edge_attr, size=(num_source, num_target))

        # Split and assign to target per type
        data_target = self.background_mesh.graph_hetero_data
        for t in node_types:
            s = target_offsets[t]
            data_target[t].phys_features = out[s, 0:self.config.num_phys_features]
            data_target[t].phys_features_spatial_d = out[s, self.config.num_phys_features:self.config.num_phys_features + self.config.num_phys_spatial_features_d]
            data_target[t].phys_features_spatial_dd = out[s, self.config.num_phys_features + self.config.num_phys_spatial_features_d:]
            data_target[t].phys_features_dt = out_dt[s]

    def transfer_physics_to_latent(self, reverse_transform_conv):
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

        out = reverse_transform_conv(x=x, edge_index=assignment, edge_attr=edge_attr, size=(num_source, num_target))

        # Assign to target (graph_mesh) base_features
        data_target = self.graph_mesh.graph_hetero_data
        for t in node_types:
            s = target_offsets[t]
            data_target[t].base_features = out[s]

    def update_BC(self, t, bc_transform_pressure, bc_transform_velocity):
        self.graph_mesh.update_bc_features(t, bc_transform_pressure, bc_transform_velocity)
        self.background_mesh.update_bc_features(t)

    def compute_base_features_dt(self, message_passing_conv):
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
        dt_dict = message_passing_conv(x_dict, data.edge_index_dict)
        
        # Assign dt
        for t in node_types:
            data[t].base_features_dt = dt_dict[t]

    def next_time_step(self, current_time, delta_t, bc_transform_pressure, bc_transform_velocity, message_passing_conv):
        # update BC
        # Update base features by integraition RK4
        # Update base features_dt by compute_base_features_dt at new time step

        # Assumes current time step is managed externally; here we advance base_features using RK4
        # Need current t; assume current_time is set (add current_time = 0.0 in _initialize if needed)
        data = self.graph_mesh.graph_hetero_data
        node_types = data.node_types
        
        # Store original base_features
        original_base = {t: data[t].base_features.clone() for t in node_types}
        
        # RK4 steps; since BC(t) changes, update BC at intermediate times
        # k1 = f(t, y)
        self.update_BC(current_time, bc_transform_pressure, bc_transform_velocity)
        self.compute_base_features_dt(message_passing_conv)
        k1 = {t: data[t].base_features_dt.clone() for t in node_types}
        
        # k2 = f(t + delta_t/2, y + delta_t/2 * k1)
        for t in node_types:
            data[t].base_features = original_base[t] + (delta_t / 2) * k1[t]
        #self.update_BC(current_time + delta_t / 2, bc_transform_pressure, bc_transform_velocity)
        self.compute_base_features_dt(message_passing_conv)
        k2 = {t: data[t].base_features_dt.clone() for t in node_types}
        
        # k3 = f(t + delta_t/2, y + delta_t/2 * k2)
        for t in node_types:
            data[t].base_features = original_base[t] + (delta_t / 2) * k2[t]
        #self.update_BC(current_time + delta_t / 2, bc_transform_pressure, bc_transform_velocity)
        self.compute_base_features_dt(message_passing_conv)
        k3 = {t: data[t].base_features_dt.clone() for t in node_types}
        
        # k4 = f(t + delta_t, y + delta_t * k3)
        for t in node_types:
            data[t].base_features = original_base[t] + delta_t * k3[t]
        #self.update_BC(current_time + delta_t, bc_transform_pressure, bc_transform_velocity)
        self.compute_base_features_dt(message_passing_conv)
        k4 = {t: data[t].base_features_dt.clone() for t in node_types}
        
        # Update: y += delta_t/6 * (k1 + 2*k2 + 2*k3 + k4)
        for t in node_types:
            data[t].base_features = original_base[t] + (delta_t / 6) * (k1[t] + 2 * k2[t] + 2 * k3[t] + k4[t])
        
        # Advance time and recompute features_dt at new time step
        self.update_BC(current_time + delta_t, bc_transform_pressure, bc_transform_velocity)
        self.compute_base_features_dt(message_passing_conv)