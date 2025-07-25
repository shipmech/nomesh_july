# conditions.py
from abc import ABC
import torch
from utils import TimeHistoryData

class BoundaryCondition(ABC):
    def __init__(self):
        pass

    def get_value(self, t: float):
        pass

    def get_derivative(self, t: float):
        pass

class SimpleBC(BoundaryCondition):
    def __init__(self, times: torch.Tensor, values: torch.Tensor):
        self.time_history = TimeHistoryData(times, values)
    
    def get_value(self, t: float):
        return self.time_history.get_value(t)
    
    def get_derivative(self, t: float):
        return self.time_history.get_derivative(t)

class PressBC(SimpleBC):
    def __init__(self, times: torch.Tensor, values: torch.Tensor):
        super().__init__(times, values)
        self.type_name = 'Press'

class VelXBC(SimpleBC):
    def __init__(self, times: torch.Tensor, values: torch.Tensor):
        super().__init__(times, values)
        self.type_name = 'VelX'

class VelYBC(SimpleBC):
    def __init__(self, times: torch.Tensor, values: torch.Tensor):
        super().__init__(times, values)
        self.type_name = 'VelY'

class InitialCondition:
    def __init__(self, u: float = 0.0, v: float = 0.0, p: float = 0.0, device=None):
        self.u = u
        self.v = v
        self.p = p
        self.device = device

    def get_initial_state(self) -> torch.Tensor:
        return torch.tensor([self.u, self.v, self.p], dtype=torch.float32, device=self.device)