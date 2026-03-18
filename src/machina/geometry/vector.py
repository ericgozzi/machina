from __future__ import annotations

from typing import Union

from machina.geometry.geometry import Geometry


class Vector(Geometry):
    def __init__(self, x: float, y: float, z: float):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z

    @property
    def data(self) -> dict:
        data = super().data
        data["type"] = "vector"
        data["x"] = self.x
        data["y"] = self.y
        data["z"] = self.z
        return data

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Vector(x, y, z)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return Vector(x, y, z)

    def __mul__(self, scalar: float):
        x = self.x * scalar
        y = self.y * scalar
        z = self.z * scalar
        return Vector(x, y, z)

    def __truediv__(self, scalar: Union[float, int]):
        x = self.x / scalar
        y = self.y / scalar
        z = self.z / scalar
        return Vector(x, y, z)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    @property
    def length(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    @classmethod
    def from_data(cls, data: dict) -> Vector:
        return cls(data["x"], data["y"], data["z"])

    def unitize(self):
        length = self.length
        self.x /= length
        self.y /= length
        self.z /= length
        return self

    def dot(self, other: Vector) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector) -> Vector:
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def transform(self, transformation):
        vector = transformation.apply_to_vector(self)
        self.x, self.y, self.z = vector.x, vector.y, vector.z
        return self
