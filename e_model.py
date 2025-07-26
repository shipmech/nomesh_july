# case.py
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import radius

def update_bc_features(graph_hetero_data,
                       graph_data,
                       t : float,
                       bc_transform_pressure = torch.nn.Identity(), bc_transform_velocity = torch.nn.Identity()):

    node_dict_data = graph_data.node_dict_data

    dict_bc_index_to_latent_value_tensor = {}

    for bc_index, bc_exact in node_dict_data.dict_BC_index_to_BC_exact.items():
        value_tensor = bc_exact.get_value(t)
        bctype_index = node_dict_data.dict_BC_index_to_BCType_index[bc_index]
        latent_value_tensor = None
        if bctype_index == 0:
            latent_value_tensor = bc_transform_pressure(value_tensor)
        elif bctype_index == 1:
            latent_value_tensor = bc_transform_velocity(value_tensor)
        
        dict_bc_index_to_latent_value_tensor[bc_index] = latent_value_tensor

    for node_type_index, bc_indices in node_dict_data.dict_type_index_to_bc_indices_list.items():
        if len(bc_indices) == 0:
            continue
        node_type_name = node_dict_data.dict_type_index_to_type_name[node_type_index]
        if node_type_name == 'Free':
            continue

        list_of_bc_values = []

        for bc_index in bc_indices:
            latent_value_tensor = dict_bc_index_to_latent_value_tensor[bc_index]
            list_of_bc_values.append(latent_value_tensor)
        
        graph_hetero_data[node_type_name].bc_features = torch.stack(list_of_bc_values, dim=0).clone()

    return graph_hetero_data

def partialy_clone_latent_graph(graph_hetero_data):
    data = HeteroData()

    for node_type_name in graph_hetero_data.node_types:
        data[node_type_name].pos = graph_hetero_data[node_type_name].pos
        data[node_type_name].base_features = graph_hetero_data[node_type_name].base_features.clone()
        data[node_type_name].base_features_dt = graph_hetero_data[node_type_name].base_features_dt.clone()
        data[node_type_name].bc_features = graph_hetero_data[node_type_name].bc_features.clone()

    #for src in graph_hetero_data.node_types:
    #        for dst in graph_hetero_data.node_types:
    #            edge_type = (src, 'influences', dst)

    for (src_type, relation, dst_type) in graph_hetero_data.edge_types:
        data[src_type, relation, dst_type].edge_index = graph_hetero_data[src_type, relation, dst_type].edge_index

    return data

def partialy_clone_background_mesh(graph_hetero_data):
    data = HeteroData()

    for node_type_name in graph_hetero_data.node_types:
        data[node_type_name].pos = graph_hetero_data[node_type_name].pos
        data[node_type_name].phys_features = graph_hetero_data[node_type_name].phys_features.clone()
        data[node_type_name].phys_features_dt = graph_hetero_data[node_type_name].phys_features_dt.clone()
        data[node_type_name].phys_features_spatial_d = graph_hetero_data[node_type_name].phys_features_spatial_d.clone()
        data[node_type_name].phys_features_spatial_dd = graph_hetero_data[node_type_name].phys_features_spatial_dd.clone()
        data[node_type_name].bc_features = graph_hetero_data[node_type_name].bc_features.clone()

    for (src_type, relation, dst_type) in graph_hetero_data.edge_types:
        data[src_type, relation, dst_type].edge_index = graph_hetero_data[src_type, relation, dst_type].edge_index

    return data

def transfer_latent_to_physics(config, latent_graph, background_mesh, transform_conv, transform_conv_dt):
    data_source = partialy_clone_latent_graph(latent_graph)
    data_target = partialy_clone_background_mesh(background_mesh)
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
        offset = offset + n

    # Step 2: Build bipartite edges (source -> target) using radius
    r = config.knn_radius_transfer  # Your radius parameter
    max_num_neighbors = 32  # Max per target; prevents explosion in dense areas
    assignment = radius(source_pos.clone(), target_pos.clone(), r, max_num_neighbors=max_num_neighbors)
    # assignment: [2, num_assignments], row 0: source idx, row 1: target idx

    # Optional: Check for isolated targets (zero neighbors) and handle (e.g., log or assign default)
    row_counts = torch.bincount(assignment[1], minlength=num_target)
    isolated = torch.where(row_counts == 0)[0]
    if len(isolated) > 0:
        print(f"Warning: {len(isolated)} target background nodes have no source latent neighbors within radius {r}.")
        # Could reselect: e.g., connect to global nearest, or skip/exclude them

    # Step 3: Compute relative positions and normalize to [0,1] by radius
    rel_pos = target_pos[assignment[1]] - source_pos[assignment[0]]
    r = config.knn_radius_transfer
    edge_attr = (rel_pos / (2 * r)) + 0.5  # Shifts [-r, r] to [0,1]; clamp if needed
    edge_attr = torch.clamp(edge_attr, 0.0, 1.0)  # Ensure bounds

    x = (source_base.clone(), None)
    x_dt = (source_base_dt.clone(), None)

    out = transform_conv(x=x, edge_index=assignment.clone(), edge_attr=edge_attr.clone(), size=(num_source, num_target))
    out_dt = transform_conv_dt(x=x_dt, edge_index=assignment.clone(), edge_attr=edge_attr.clone(), size=(num_source, num_target))

    # Split and assign to target per type
    data_target = partialy_clone_background_mesh(background_mesh)
    for t in node_types:
        s = target_offsets[t]
        data_target[t].phys_features = out[s, 0:config.num_phys_features].clone()
        data_target[t].phys_features_spatial_d = out[s, config.num_phys_features:config.num_phys_features + config.num_phys_spatial_features_d].clone()
        data_target[t].phys_features_spatial_dd = out[s, config.num_phys_features + config.num_phys_spatial_features_d:].clone()
        data_target[t].phys_features_dt = out_dt[s].clone()

    return data_target

def transfer_physics_to_latent(config, latent_graph, background_mesh, reverse_transform_conv):
    data_source = partialy_clone_background_mesh(background_mesh)  # Now background is source (physics)
    data_target = partialy_clone_latent_graph(latent_graph) # Graph is target (latents)
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
        offset = offset + n

    # Step 2: Build bipartite edges (source -> target) using radius
    r = config.knn_radius_transfer  # Same radius
    max_num_neighbors = 32  # Same max
    assignment = radius(source_pos.clone(), target_pos.clone(), r, max_num_neighbors=max_num_neighbors)
    # assignment: [2, num_assignments], row 0: source idx (background), row 1: target idx (graph)

    # Optional: Check for isolated targets
    row_counts = torch.bincount(assignment[1], minlength=num_target)
    isolated = torch.where(row_counts == 0)[0]
    if len(isolated) > 0:
        print(f"Warning: {len(isolated)} target latent nodes have no source background neighbors within radius {r}.")

    # Step 3: Compute relative positions and normalize to [0,1] by radius
    rel_pos = target_pos[assignment[1]] - source_pos[assignment[0]]
    edge_attr = (rel_pos / (2 * r)) + 0.5
    edge_attr = torch.clamp(edge_attr, 0.0, 1.0)

    x = (source_phys.clone(), None)

    out = reverse_transform_conv(x=x, edge_index=assignment.clone(), edge_attr=edge_attr.clone(), size=(num_source, num_target))

    # Assign to target (graph_mesh) base_features
    data_target = partialy_clone_latent_graph(latent_graph)
    for t in node_types:
        s = target_offsets[t]
        data_target[t].base_features = out[s].clone()

    return data_target

def compute_base_features_dt(latent_graph, message_passing_conv, output = 'latent_graph'):
    #print(latent_graph)
    data = partialy_clone_latent_graph(latent_graph)

    node_types = data.node_types  # ['Free', 'Press', 'NoSlip']
    
    # Prepare x_dict: cat(base_features, bc_features) per node type
    # Note: bc_features assumed to be latent
    x_dict = {}
    for t in node_types:
        base = data[t].base_features.clone()
        bc = data[t].bc_features.clone()
        x_dict[t] = torch.cat([base, bc], dim=-1) if bc.size(1) > 0 else base
    
    # Run message passing
    dt_dict = message_passing_conv(x_dict, data.edge_index_dict)
    
    if output == 'dt':
        return dt_dict

    for t in node_types:
        data[t].base_features_dt = dt_dict[t].clone()

    return data

def next_time_step(latent_graph, graph_data, current_time, delta_t, bc_transform_pressure, bc_transform_velocity, message_passing_conv):
    # update BC
    # Update base features by integraition RK4
    # Update base features_dt by compute_base_features_dt at new time step

    latent_graph = update_bc_features(latent_graph, graph_data, current_time, bc_transform_pressure, bc_transform_velocity)

    node_types = latent_graph.node_types
    #original_base = {t: latent_graph[t].base_features for t in node_types}
    
    # RK4 steps; since BC(t) changes, update BC at intermediate times
    # k1 = f(t, y)
    #data_rk4 = partialy_clone_latent_graph(initial_data)   # maybe latent_graph is better
    k1 = compute_base_features_dt(latent_graph, message_passing_conv, output='dt')
    #k1 = {t: data1[t].base_features_dt.clone() for t in node_types}
    
    # k2 = f(t + delta_t/2, y + delta_t/2 * k1)
    k2_graph = partialy_clone_latent_graph(latent_graph)
    for t in node_types:
        k2_graph[t].base_features = latent_graph[t].base_features + 0.5 * delta_t * k1[t]
    k2 = compute_base_features_dt(k2_graph, message_passing_conv, output='dt')
    
    # k3 = f(t + delta_t/2, y + delta_t/2 * k2)
    k3_graph = partialy_clone_latent_graph(latent_graph)
    for t in node_types:
         k3_graph[t].base_features = latent_graph[t].base_features + 0.5 * delta_t * k2[t]
    k3 = compute_base_features_dt(k3_graph, message_passing_conv, output='dt')
    
    # k4 = f(t + delta_t, y + delta_t * k3)
    k4_graph = partialy_clone_latent_graph(latent_graph)
    for t in node_types:
        k4_graph[t].base_features = latent_graph[t].base_features + delta_t * k3[t]
    k4 = compute_base_features_dt(k4_graph, message_passing_conv, output='dt')
    
    # Update: y += delta_t/6 * (k1 + 2*k2 + 2*k3 + k4)
    for t in node_types:
        latent_graph[t].base_features = latent_graph[t].base_features + (delta_t / 6) * (k1[t] + 2 * k2[t] + 2 * k3[t] + k4[t])

    latent_graph = compute_base_features_dt(latent_graph, message_passing_conv)

    return latent_graph