from __future__ import annotations

from typing import TYPE_CHECKING

from .geometry import Geometry
from .vector import Vector

if TYPE_CHECKING:
    from .transformation.transformation import Transformation


class Point(Geometry):
    def __init__(self, x: float, y: float, z: float, **kwargs):
        super().__init__(**kwargs)
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

    def __str__(self) -> str:
        return f"Point({self.x}, {self.y}, {self.z})"

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y}, z={self.z})"

    def __add__(self, other) -> Point:
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Point(x, y, z)

    def __sub__(self, other) -> Vector:
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __getitem__(self, index) -> float:
        return (self.x, self.y, self.z)[index]

    def __setitem__(self, index, value: float) -> None:
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError("Point index out of range")

    def __len__(self) -> int:
        return 3

    @classmethod
    def from_data(cls, data: dict) -> Point:
        return cls(data["x"], data["y"], data["z"])

    def transform(self, transformation: Transformation) -> Point:
        point = transformation.apply_to_point(self)
        self.x, self.y, self.z = point.x, point.y, point.z
        return self

    def distance_to(self, other: Point) -> float:
        return (
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        ) ** 0.5
