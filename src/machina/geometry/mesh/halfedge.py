from __future__ import annotations

from typing import TYPE_CHECKING

from machina.geometry.geometry import Geometry

if TYPE_CHECKING:
    from .face import Face
    from .vertex import Vertex


class Halfedge(Geometry):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vertex: Vertex
        self.twin: Halfedge
        self.next: Halfedge
        self.face: Face

    @property
    def data(self) -> dict:
        data = super().data
        data["vertex"] = self.vertex.data
        data["twin"] = self.twin.data
        data["next"] = self.next.data
        data["face"] = self.face.data
        return data

    @classmethod
    def from_data(cls, data: dict) -> Halfedge:
        raise NotImplementedError
