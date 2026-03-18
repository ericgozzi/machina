from __future__ import annotations

from typing import Optional

from machina.geometry.frame import Frame
from machina.geometry.geometry import Geometry
from machina.geometry.point import Point


class Cuboid(Geometry):
    def __init__(
        self,
        x_size: float,
        y_size: float,
        z_size: float,
        frame: Optional[Frame],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.frame = frame or Frame.world()

    @property
    def data(self) -> dict:
        data = super().data
        data["type"] = "cuboid"
        data["x_size"] = self.x_size
        data["y_size"] = self.y_size
        data["z_size"] = self.z_size
        data["frame"] = self.frame.data
        return data

    @property
    def volume(self) -> float:
        return self.x_size * self.y_size * self.z_size

    @property
    def vertices(self) -> list[Point]:
        v0 = (
            self.frame.origin
            + self.frame.x_axis * (-self.x_size / 2)
            + self.frame.y_axis * (-self.y_size / 2)
            + self.frame.z_axis * (-self.z_size / 2)
        )
        v1 = (
            self.frame.origin
            + self.frame.x_axis * (self.x_size / 2)
            + self.frame.y_axis * (-self.y_size / 2)
            + self.frame.z_axis * (-self.z_size / 2)
        )
        v2 = (
            self.frame.origin
            + self.frame.x_axis * (self.x_size / 2)
            + self.frame.y_axis * (self.y_size / 2)
            + self.frame.z_axis * (-self.z_size / 2)
        )
        v3 = (
            self.frame.origin
            + self.frame.x_axis * (-self.x_size / 2)
            + self.frame.y_axis * (self.y_size / 2)
            + self.frame.z_axis * (-self.z_size / 2)
        )
        v4 = (
            self.frame.origin
            + self.frame.x_axis * (-self.x_size / 2)
            + self.frame.y_axis * (-self.y_size / 2)
            + self.frame.z_axis * (self.z_size / 2)
        )
        v5 = (
            self.frame.origin
            + self.frame.x_axis * (self.x_size / 2)
            + self.frame.y_axis * (-self.y_size / 2)
            + self.frame.z_axis * (self.z_size / 2)
        )
        v6 = (
            self.frame.origin
            + self.frame.x_axis * (self.x_size / 2)
            + self.frame.y_axis * (self.y_size / 2)
            + self.frame.z_axis * (self.z_size / 2)
        )
        v7 = (
            self.frame.origin
            + self.frame.x_axis * (-self.x_size / 2)
            + self.frame.y_axis * (self.y_size / 2)
            + self.frame.z_axis * (self.z_size / 2)
        )
        return [v0, v1, v2, v3, v4, v5, v6, v7]

    @classmethod
    def from_data(cls, data):
        return cls(
            data["x_size"],
            data["y_size"],
            data["z_size"],
            frame=Frame.from_data(data["frame"]),
        )

    def transform(self, transformation):
        self.frame.transform(transformation)

        self.x_size *= transformation.scale_x
        self.y_size *= transformation.scale_y
        self.z_size *= transformation.scale_z

        return self
