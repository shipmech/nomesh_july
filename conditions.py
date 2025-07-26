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
        u = self.time_history_u.get_value(t)
        v = self.time_history_v.get_value(t)
        return torch.tensor([u, v], dtype=u.dtype, device=u.device)
    
    def get_derivative(self, t: float):
        dudt = self.time_history_u.get_derivative(t)
        dvdt = self.time_history_v.get_derivative(t)
        return torch.tensor([dudt, dvdt], dtype=dudt.dtype, device=dudt.device)