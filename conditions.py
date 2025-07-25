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

class PressureBC(BoundaryCondition):
    def __init__(self, times: torch.Tensor, values: torch.Tensor):
        self.time_history = TimeHistoryData(times, values)
        self.type_name = 'Press'

    def get_value(self, t: float):
        return self.time_history.get_value(t)
    
    def get_derivative(self, t: float):
        return self.time_history.get_derivative(t)

class VelocityBC(BoundaryCondition):
    def __init__(self,
                 times_u: torch.Tensor, values_u: torch.Tensor,
                 times_v: torch.Tensor, values_v: torch.Tensor):
        self.time_history_u = TimeHistoryData(times_u, values_u)
        self.time_history_v = TimeHistoryData(times_v, values_v)
        self.type_name = 'Vel'

    def get_value(self, t: float):
        return self.time_history_u.get_value(t), self.time_history_v.get_value(t)
    
    def get_derivative(self, t: float):
        return self.time_history_u.get_derivative(t), self.time_history_v.get_derivative(t)

class InitialCondition:
    def __init__(self, u: float = 0.0, v: float = 0.0, p: float = 0.0, device=None):
        self.u = u
        self.v = v
        self.p = p
        self.device = device

    def get_initial_state(self) -> torch.Tensor:
        return torch.tensor([self.u, self.v, self.p], dtype=torch.float32, device=self.device)