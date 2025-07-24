# conditions.py
from abc import ABC
import torch
from utils import TimeHistoryData

class BoundaryCondition(ABC):
    def __init__(self, time_history: TimeHistoryData):
        self.time_history = time_history

    def get_value(self, t: float) -> torch.Tensor:
        return self.time_history.get_value(t)

class PressureBC(BoundaryCondition):
    def get_pressure(self, t: float) -> torch.Tensor:
        return self.get_value(t)

class VelocityBC(BoundaryCondition):
    def get_velocity(self, t: float) -> torch.Tensor:
        return self.get_value(t)

class InitialCondition:
    def __init__(self, u: float = 0.0, v: float = 0.0, p: float = 0.0, device=None):
        self.u = u
        self.v = v
        self.p = p
        self.device = device

    def get_initial_state(self) -> torch.Tensor:
        return torch.tensor([self.u, self.v, self.p], dtype=torch.float32, device=self.device)