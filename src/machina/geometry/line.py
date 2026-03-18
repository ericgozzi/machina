from __future__ import annotations

from machina.geometry.geometry import Geometry
from machina.geometry.point import Point
from machina.geometry.vector import Vector


class Line(Geometry):
    def __init__(self, start: Point, end: Point):
        super().__init__()
        self.start = start
        self.end = end

    @property
    def data(self) -> dict:
        data = super().data
        data["type"] = "line"
        data["start"] = self.start.data
        data["end"] = self.end.data
        return data

    @property
    def length(self) -> float:
        length = (
            (self.end.x - self.start.x) ** 2
            + (self.end.y - self.start.y) ** 2
            + (self.end.z - self.start.z) ** 2
        ) ** 0.5
        return length

    @property
    def direction(self) -> Vector:
        direction = Vector(
            self.end.x - self.start.x,
            self.end.y - self.start.y,
            self.end.z - self.start.z,
        )
        return direction

    @property
    def midpoint(self) -> Point:
        midpoint = Point(
            (self.start.x + self.end.x) / 2,
            (self.start.y + self.end.y) / 2,
            (self.start.z + self.end.z) / 2,
        )
        return midpoint

    @classmethod
    def from_data(cls, data: dict):
        return cls(Point.from_data(data["start"]), Point.from_data(data["end"]))

    def transform(self, transformation):
        self.start.transform(transformation)
        self.end.transform(transformation)
        return self
