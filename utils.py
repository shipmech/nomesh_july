# utils.py
import torch

class TimeHistoryData:
    def __init__(self, times: torch.Tensor, values: torch.Tensor):
        self.times = times
        self.values = values
        self.derivatives = self.compute_derivatives()

    def compute_derivatives(self) -> torch.Tensor:
        if len(self.times) < 2:
            return torch.zeros_like(self.values)

        derivatives = torch.zeros_like(self.values)
        # Forward difference for first point
        derivatives[0] = (self.values[1] - self.values[0]) / (self.times[1] - self.times[0])
        # Backward difference for last point
        derivatives[-1] = (self.values[-1] - self.values[-2]) / (self.times[-1] - self.times[-2])
        # Central differences for interior points
        for i in range(1, len(self.times) - 1):
            dt_prev = self.times[i] - self.times[i - 1]
            dt_next = self.times[i + 1] - self.times[i]
            derivatives[i] = ((self.values[i] - self.values[i - 1]) / dt_prev + (self.values[i + 1] - self.values[i]) / dt_next) / 2
        return derivatives

    def get_value(self, t: float) -> torch.Tensor:
        t_tensor = torch.tensor(t, dtype=self.times.dtype, device=self.times.device)
        idx = torch.searchsorted(self.times, t_tensor)
        if idx == 0:
            return self.values[0]
        if idx == len(self.times):
            return self.values[-1]
        # Linear interpolation
        t1, t2 = self.times[idx - 1], self.times[idx]
        v1, v2 = self.values[idx - 1], self.values[idx]
        return v1 + (v2 - v1) * (t_tensor - t1) / (t2 - t1)

    def get_derivative(self, t: float) -> torch.Tensor:
        t_tensor = torch.tensor(t, dtype=self.times.dtype, device=self.times.device)
        idx = torch.searchsorted(self.times, t_tensor)
        if idx == 0:
            return self.derivatives[0]
        if idx == len(self.times):
            return self.derivatives[-1]
        # Linear interpolation
        t1, t2 = self.times[idx - 1], self.times[idx]
        d1, d2 = self.derivatives[idx - 1], self.derivatives[idx]
        return d1 + (d2 - d1) * (t_tensor - t1) / (t2 - t1)