from __future__ import annotations

import math

from machina.geometry.geometry import Geometry
from machina.geometry.point import Point


class Sphere(Geometry):
    def __init__(self, center: Point, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.center = center
        self.radius = radius

    @property
    def data(self) -> dict:
        data = super().data
        data["type"] = "sphere"
        data["center"] = self.center.data
        data["radius"] = self.radius
        return data

    @property
    def volume(self) -> float:
        return 4 / 3 * math.pi * self.radius**3

    @classmethod
    def from_data(cls, data: dict) -> Sphere:
        return cls(Point.from_data(data["center"]), data["radius"])

    def transform(self, transformation):
        self.center.transform(transformation)
        self.radius *= transformation.scale_factor
