from __future__ import annotations

from machina.geometry.geometry import Geometry
from machina.geometry.point import Point
from machina.geometry.vector import Vector


class Frame(Geometry):
    def __init__(self, origin: Point, x_axis: Vector, y_axis: Vector, **kwargs):
        super().__init__(**kwargs)
        self.origin = origin
        self.x_axis = x_axis
        self.y_axis = y_axis

    @property
    def data(self) -> dict:
        return {
            "type": "frame",
            "origin": self.origin.data,
            "x_axis": self.x_axis.data,
            "y_axis": self.y_axis.data,
        }

    @property
    def z_axis(self) -> Vector:
        return self.x_axis.cross(self.y_axis)

    @classmethod
    def from_data(cls, data: dict) -> Frame:
        return cls(
            origin=Point.from_data(data["origin"]),
            x_axis=Vector.from_data(data["x_axis"]),
            y_axis=Vector.from_data(data["y_axis"]),
        )

    @classmethod
    def world(cls) -> Frame:
        return cls(Point(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))

    def transform(self, transformation):
        self.origin.transform(transformation)
        self.x_axis.transform(transformation)
        self.y_axis.transform(transformation)
        return self
