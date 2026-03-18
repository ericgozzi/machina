from __future__ import annotations

from machina.geometry.geometry import Geometry
from machina.geometry.vector import Vector


class Point(Geometry):
    def __init__(self, x: float, y: float, z: float):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z

    @property
    def data(self) -> dict:
        data = super().data
        data["type"] = "point"
        data["x"] = self.x
        data["y"] = self.y
        data["z"] = self.z
        return data

    def __repr__(self):
        return f"Point({self.x}, {self.y}, {self.z})"

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Point(x, y, z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    @classmethod
    def from_data(cls, data: dict) -> Point:
        return cls(data["x"], data["y"], data["z"])

    def transform(self, transformation):
        point = transformation.apply_to_point(self)
        self.x, self.y, self.z = point.x, point.y, point.z
        return self
