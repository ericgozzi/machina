from __future__ import annotations

from ..vector import Vector
from .transformation import Transformation


class Translation(Transformation):
    def __init__(self, x: float, y: float, z: float):
        matrix = [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
        super().__init__(matrix)

    @property
    def x(self) -> float:
        return self.matrix[0][3]

    @x.setter
    def x(self, value: float):
        self.matrix[0][3] = value

    @property
    def y(self) -> float:
        return self.matrix[1][3]

    @y.setter
    def y(self, value: float):
        self.matrix[1][3] = value

    @property
    def z(self) -> float:
        return self.matrix[2][3]

    @z.setter
    def z(self, value: float):
        self.matrix[2][3] = value

    @property
    def vector(self):
        return Vector(self.x, self.y, self.z)

    @classmethod
    def from_vector(cls, vector: Vector) -> Translation:
        return cls(vector.x, vector.y, vector.z)
