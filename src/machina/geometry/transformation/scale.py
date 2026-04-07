from __future__ import annotations

from ..point import Point
from .transformation import Transformation


class Scale(Transformation):
    def __init__(self, x_factor: float, y_factor: float, z_factor: float, point: Point):
        """
        Scales geometry relative to a specific pivot point.
        """
        # 1. Scaling factors stay on the diagonal
        s00 = x_factor
        s11 = y_factor
        s22 = z_factor

        # 2. Calculate translation to keep the pivot point fixed
        # Formula: Point - (Scale_Factor * Point)
        tx = point.x - (s00 * point.x)
        ty = point.y - (s11 * point.y)
        tz = point.z - (s22 * point.z)

        self.matrix = [
            [s00, 0, 0, tx],
            [0, s11, 0, ty],
            [0, 0, s22, tz],
            [0, 0, 0, 1],
        ]
        super().__init__(self.matrix)
