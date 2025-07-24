import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from case import Case  # New import

class FluidDynamicsDataset(Dataset):
    def __init__(self, config, is_training: bool = True):
        self.config = config
        self.is_training = is_training
        self.cases = self._generate_cases()

    def _generate_cases(self) -> list[Case]:
        if not self.is_training:
            # Fixed validation case
            fixed_config = self.config  # Copy and override
            fixed_config.domain_width_range = (1.0, 1.0)
            fixed_config.domain_height_range = (0.5, 0.5)
            fixed_config.inlet_velocity_range = (0.5, 0.5)
            fixed_config.outlet_pressure_range = (100.0, 100.0)
            fixed_config.num_nodes_range = (1500, 1500)  # Fixed 1500 nodes
            return [Case(fixed_config)]
        
        # Training: random cases
        return [Case(self.config) for _ in range(self.config.num_training_cases)]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Case:
        return self.cases[idx]

class FluidDynamicsDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset: FluidDynamicsDataset | None = None
        self.val_dataset: FluidDynamicsDataset | None = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_dataset = FluidDynamicsDataset(self.config, is_training=True)
            self.val_dataset = FluidDynamicsDataset(self.config, is_training=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Placeholder for Fenics (to be added later)
    def load_fenics_data(self) -> dict[str, torch.Tensor]:
        pass  # Implement in future steps