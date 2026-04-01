from __future__ import annotations

from typing import TYPE_CHECKING

from machina.geometry.geometry import Geometry

if TYPE_CHECKING:
    from machina.geometry.halfedge import Halfedge
    from machina.geometry.vertex import Vertex


class Face(Geometry):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.halfedge: Halfedge = None
        self.index: int = None

    @property
    def data(self) -> dict:
        data = super().data
        data["vertices_indices"] = [v.index for v in self.vertices]
        data["index"] = self.index
        return data

    @property
    def vertices(self) -> list[Vertex]:
        vertices = []
        halfedge_start = self.halfedge
        halfedge = self.halfedge
        while True:
            vertices.append(halfedge.vertex)
            halfedge = halfedge.next
            if halfedge == halfedge_start:
                break
        return vertices

    def halfedges(self) -> list[Halfedge]:
        halfedges = []
        halfedge_start = self.halfedge
        halfedge = self.halfedge
        while True:
            halfedges.append(halfedge)
            halfedge = halfedge.next
            if halfedge == halfedge_start:
                break
        return halfedges

    @classmethod
    def from_data(cls, data: dict) -> Face:
        pass
