from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from machina.geometry.point import Point
    from machina.geometry.vector import Vector

    from .transformation.transformation import Transformation


class Geometry(ABC):
    def __init__(self, **kwargs):
        self.attributes = {}
        self.attributes.update(kwargs)

    @property
    def data(self) -> dict:
        data = {}
        for key, value in self.attributes.items():
            data[key] = value
        return data

    @classmethod
    @abstractmethod
    def from_data(cls, data):
        raise NotImplementedError

    def copy(self):
        data = self.data
        return type(self).from_data(data)

    # ---- TRANSFORM

    def transform(self, transformation: Transformation):
        raise NotImplementedError

    def rotate(self, angle: float, axis: Vector, point: Point):
        raise NotImplementedError

    def translate(self, vector: Vector):
        raise NotImplementedError

    def scale(self, scale_x: float, scale_y: float, scale_z: float):
        raise NotImplementedError
