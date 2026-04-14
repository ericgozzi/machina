from __future__ import annotations

import math

from ..point import Point
from ..vector import Vector
from .transformation import Transformation


class Rotation(Transformation):
    def __init__(self, angle: float, axis: Vector, point: Point):
        axis = axis.copy()
        axis.unitize()

        c = math.cos(angle)
        s = math.sin(angle)
        t = 1 - c

        # rotation indices
        r00 = t * axis.x**2 + c
        r01 = t * axis.x * axis.y - s * axis.z
        r02 = t * axis.x * axis.z + s * axis.y

        r10 = t * axis.x * axis.y + s * axis.z
        r11 = t * axis.y**2 + c
        r12 = t * axis.y * axis.z - s * axis.x

        r20 = t * axis.x * axis.z - s * axis.y
        r21 = t * axis.y * axis.z + s * axis.x
        r22 = t * axis.z**2 + c

        # translation indices
        tx = point.x - (r00 * point.x + r01 * point.y + r02 * point.z)
        ty = point.y - (r10 * point.x + r11 * point.y + r12 * point.z)
        tz = point.z - (r20 * point.x + r21 * point.y + r22 * point.z)

        self.matrix = [
            [r00, r01, r02, tx],
            [r10, r11, r12, ty],
            [r20, r21, r22, tz],
            [0, 0, 0, 1],
        ]
        super().__init__(self.matrix)
