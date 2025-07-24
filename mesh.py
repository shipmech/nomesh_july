import torch
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from config import SimulationConfig

def create_meshes(graph_data: Data, config: SimulationConfig):
    graph_data = update_edges(graph_data, config)
    
    # Convert width and height to scalars
    width = graph_data.width.item() if torch.is_tensor(graph_data.width) else graph_data.width
    height = graph_data.height.item() if torch.is_tensor(graph_data.height) else graph_data.height
    
    x = torch.linspace(0, width, 50, device=config.device)
    y = torch.linspace(0, height, 50, device=config.device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    target_points = torch.stack([X.ravel(), Y.ravel()], axis=1)
    
    return graph_data, target_points

def update_edges(graph_data: Data, config: SimulationConfig) -> Data:
    edge_index = knn_graph(graph_data.pos, k=config.k_neighbors, flow="source_to_target")
    dist = torch.norm(graph_data.pos[edge_index[1]] - graph_data.pos[edge_index[0]], dim=1)
    edge_index = edge_index[:, dist <= config.radius]
    graph_data.edge_index = edge_index
    return graph_data