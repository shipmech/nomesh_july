import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import lightning as L
import numpy as np
from config import SimulationConfig

class FluidDynamicsDataset:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.cases = self._generate_cases()

    def _generate_cases(self) -> list[Data]:
        cases = []
        # Training cases (random)
        for i in range(2):  # Reduced to 2 training cases
            width = np.random.uniform(*self.config.domain_width_range)
            height = np.random.uniform(*self.config.domain_height_range)
            num_nodes = np.random.randint(*self.config.num_nodes_range)
            inlet_u = np.random.uniform(*self.config.inlet_velocity_range)
            outlet_p = np.random.uniform(*self.config.outlet_pressure_range)
            graph_data = self._create_graph(width, height, num_nodes, inlet_u, 0.0, outlet_p)
            graph_data.case_id = f"training_{i+1}"
            cases.append(graph_data)

        # Validation case (fixed, aligned with training domain)
        graph_data = self._create_graph(
            np.mean(self.config.domain_width_range),  # Midpoint of training width range
            np.mean(self.config.domain_height_range),  # Midpoint of training height range
            self.config.validation_num_nodes,
            inlet_u=0.5,
            inlet_v=0.0,
            outlet_p=100.0
        )
        graph_data.case_id = "validation_1"
        cases.append(graph_data)
        return cases

    def _create_graph(self, width: float, height: float, num_nodes: int, inlet_u: float, inlet_v: float, outlet_p: float) -> Data:
        if self.config.distribution_type == "uniform":
            # Generate a complete grid and use all nodes up to num_nodes
            grid_size = int(np.ceil(np.sqrt(num_nodes)))
            while grid_size * grid_size < num_nodes:
                grid_size += 1
            x = np.linspace(0, width, grid_size)
            y = np.linspace(0, height, grid_size)
            X, Y = np.meshgrid(x, y, indexing='ij')
            pos = np.stack([X.ravel(), Y.ravel()], axis=1)
            pos = pos[:num_nodes]  # Take exactly num_nodes, ensuring boundary nodes are included
        else:
            pos = np.random.uniform([0, 0], [width, height], (num_nodes, 2))

        pos = torch.tensor(pos, dtype=torch.float32, device=self.config.device)  # Explicitly set to float32

        node_type = torch.zeros(num_nodes, dtype=torch.long, device=self.config.device)
        boundary_data = torch.zeros(num_nodes, max(self.config.number_of_pressure_bc_latent_features, self.config.number_of_velocities_bc_latent_features), device=self.config.device)

        # Assign boundary conditions with explicit checks
        for i, p in enumerate(pos):
            if torch.isclose(p[0], torch.tensor(0.0, dtype=torch.float32, device=self.config.device), atol=0.005):  # Inlet (left)
                node_type[i] = 2
                boundary_data[i, :2] = torch.tensor([inlet_u, inlet_v], dtype=torch.float32, device=self.config.device)
            elif torch.isclose(p[0], torch.tensor(width, dtype=torch.float32, device=self.config.device), atol=0.005):  # Outlet (right)
                node_type[i] = 1
                boundary_data[i, :1] = torch.tensor([outlet_p], dtype=torch.float32, device=self.config.device)
            elif torch.isclose(p[1], torch.tensor(0.0, dtype=torch.float32, device=self.config.device), atol=0.005) or \
                 torch.isclose(p[1], torch.tensor(height, dtype=torch.float32, device=self.config.device), atol=0.005):  # Top and bottom
                node_type[i] = 2
                boundary_data[i, :2] = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.config.device)

        x = torch.zeros(num_nodes, self.config.number_of_base_latent_features + max(self.config.number_of_pressure_bc_latent_features, self.config.number_of_velocities_bc_latent_features) + 6, device=self.config.device)

        return Data(
            pos=pos,
            x=x,
            node_type=node_type,
            boundary_data=boundary_data,
            width=torch.tensor(width, dtype=torch.float32, device=self.config.device),
            height=torch.tensor(height, dtype=torch.float32, device=self.config.device),
            distribution_type=self.config.distribution_type
        )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Data:
        return self.cases[idx]

class FluidDynamicsDataModule(L.LightningDataModule):
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.dataset = None

    def setup(self, stage: str = None):
        self.dataset = FluidDynamicsDataset(self.config)
        self.train_dataset = [self.dataset[i] for i in range(2)]
        self.val_dataset = [self.dataset[-1]]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=0, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False)