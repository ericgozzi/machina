from __future__ import annotations

import math

from machina.geometry.geometry import Geometry
from machina.geometry.point import Point
from machina.geometry.vector import Vector


class Curve(Geometry):
    def __init__(self, points: list[Point], **kwargs):
        super().__init__(**kwargs)
        self.points = points

    @property
    def data(self) -> dict:
        data = super().data
        data["type"] = "curve"
        data["points"] = [p.data for p in self.points]
        return data

    @property
    def points(self) -> list[Point]:
        return self._points

    @points.setter
    def points(self, value: list[Point]):
        if len(value) < 2:
            raise ValueError("A curve must have at least 2 points")
        self._points = value

    @property
    def degree(self) -> int:
        return len(self.points) - 1

    @property
    def midpoint(self) -> Point:
        return self.point_at(0.5)

    @classmethod
    def from_data(cls, data: dict) -> Curve:
        points = [Point.from_data(p_data) for p_data in data["points"]]
        return cls(points)

    def point_at(self, t: float) -> Point:
        """
        Calculates a single point at time t (0.0 to 1.0)
        using the explicit Bernstein polynomial formula.
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError("t must be between 0.0 and 1.0")

        n = self.degree
        res_x = 0.0
        res_y = 0.0
        res_z = 0.0  # Assuming 3D, remove if only 2D

        for i, p in enumerate(self.points):
            # Calculate the Bernstein basis polynomial: B_{i,n}(t)
            # Formula: comb(n, i) * (1-t)**(n-i) * t**i
            basis = math.comb(n, i) * ((1 - t) ** (n - i)) * (t**i)

            res_x += p.x * basis
            res_y += p.y * basis
            res_z += p.z * basis

        return Point(res_x, res_y, res_z)

    def tangent_at(self, t: float) -> Vector:
        """
        Calculates the tangent vector (velocity) at time t (0.0 to 1.0).
        Returns a Point object representing the directional vector.
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError("t must be between 0.0 and 1.0")

        n = self.degree
        if n < 1:
            return Vector(0, 0, 0)  # A single point has no tangent

        # The derivative is a curve of degree n-1
        derivative_degree = n - 1
        res_x = 0.0
        res_y = 0.0
        res_z = 0.0

        for i in range(n):
            # Calculate the difference between adjacent control points
            # This represents the direction of the control polygon segments
            p_current = self.points[i]
            p_next = self.points[i + 1]

            diff_x = p_next.x - p_current.x
            diff_y = p_next.y - p_current.y
            diff_z = p_next.z - p_current.z

            # Bernstein basis for the derivative curve (degree n-1)
            basis = (
                math.comb(derivative_degree, i)
                * ((1 - t) ** (derivative_degree - i))
                * (t**i)
            )

            res_x += diff_x * basis
            res_y += diff_y * basis
            res_z += diff_z * basis

        # Scale by the original degree n
        return Vector(res_x * n, res_y * n, res_z * n)
