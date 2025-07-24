from abc import ABC, abstractmethod
import torch
from utils import TimeHistoryData  # Assuming utils.py is in the same directory

class BoundaryCondition(ABC):
    def __init__(self, time_history: TimeHistoryData):
        self.time_history = time_history

    @abstractmethod
    def get_value(self, t: float) -> torch.Tensor:
        pass

class PressureBC(BoundaryCondition):
    def get_value(self, t: float) -> torch.Tensor:
        return self.time_history.get_value(t)

class VelocityBC(BoundaryCondition):
    def get_value(self, t: float) -> torch.Tensor:
        return self.time_history.get_value(t)

class InitialCondition:
    def __init__(self, u: float = 0.0, v: float = 0.0, p: float = 0.0):
        self.u = u
        self.v = v
        self.p = p

    def get_initial_state(self) -> torch.Tensor:
        return torch.tensor([self.u, self.v, self.p], dtype=torch.float32)