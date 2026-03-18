from typing import Optional

from machina.geometry.point import Point


class Vertex(Point):
    def __init__(
        self, x: float, y: float, z: float, index: Optional[int] = None, **kwargs
    ):
        super().__init__(x, y, z, index=index, **kwargs)
        self.halfedge: Halfedge = None
        self.index = index

    @property
    def data(self) -> dict:
        data = super().data
        data["index"] = self.index
        data["attributes"] = self.attributes
        return data

    @classmethod
    def from_data(cls, data: dict):
        return cls(data["x"], data["y"], data["z"], index=data["index"])
