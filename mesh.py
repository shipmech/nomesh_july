# mesh.py
import torch
import pymesh
from torch_geometric.nn import knn_graph
from geometry import Geometry

class Mesh:
    def __init__(self, pymesh_mesh: pymesh.Mesh, boundary_node_sets: dict[str, set[int]], device):
        self.pymesh_mesh = pymesh_mesh
        self.boundary_node_sets = boundary_node_sets
        self.device = device
        self.nodes = torch.tensor(pymesh_mesh.vertices, dtype=torch.float32, device=device)

    def get_nodes(self) -> torch.Tensor:
        return self.nodes

    def get_connectivity(self) -> torch.Tensor:
        return torch.tensor(self.pymesh_mesh.faces, dtype=torch.int64, device=self.device)

    def exclude_obstacles(self, obstacles: list[Geometry]):
        nodes = self.nodes
        keep_mask = torch.ones(nodes.shape[0], dtype=torch.bool, device=self.device)
        for obs in obstacles:
            if isinstance(obs, Sphere):
                dist = torch.norm(nodes - torch.tensor(obs.center, dtype=torch.float32, device=self.device), dim=1)
                keep_mask &= (dist > obs.radius)
        # Update PyMesh mesh (simplified; real impl would remesh)
        new_vertices = nodes[keep_mask].cpu().numpy()
        self.pymesh_mesh = pymesh.form_mesh(new_vertices, self.pymesh_mesh.faces)  # Faces may need recalculation
        self.nodes = nodes[keep_mask]

    def restructure(self):
        # Placeholder for mesh restructuring (e.g., refinement)
        pass

def update_edges(pos: torch.Tensor, k: int, radius: float) -> torch.Tensor:
    edge_index = knn_graph(pos, k=k)
    dist = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
    mask = dist < radius
    return edge_index[:, mask]