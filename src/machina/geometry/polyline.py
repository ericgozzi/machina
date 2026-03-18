from __future__ import annotations

from machina.geometry.geometry import Geometry
from machina.geometry.point import Point


class Polyline(Geometry):
    def __init__(self, points: list[Point], **kwargs):
        super().__init__(**kwargs)
        self.points = points

    @property
    def data(self) -> dict:
        data = super().data
        data["type"] = "polyline"
        data["points"] = [point.data for point in self.points]
        return data

    @property
    def is_closed(self):
        if self.points[0] == self.points[-1]:
            return True
        return False

    @classmethod
    def from_data(cls, data):
        points = [Point.from_data(point_data) for point_data in data["points"]]
        return cls(points)

    def close(self) -> Polyline:
        if not self.is_closed:
            self.points.append(self.points[0])
        return self

    # ---- TRANSFORMATIONS
