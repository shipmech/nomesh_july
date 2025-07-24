from abc import ABC, abstractmethod

class Geometry(ABC):
    @abstractmethod
    def get_bounds(self) -> tuple[float, float, float, float]:
        pass

class Box(Geometry):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def get_bounds(self) -> tuple[float, float, float, float]:
        return 0.0, self.width, 0.0, self.height

class Sphere(Geometry):
    def __init__(self, radius: float, center: tuple[float, float]):
        self.radius = radius
        self.center = center

    def get_bounds(self) -> tuple[float, float, float, float]:
        # Bounding box for sphere
        cx, cy = self.center
        return cx - self.radius, cx + self.radius, cy - self.radius, cy + self.radius