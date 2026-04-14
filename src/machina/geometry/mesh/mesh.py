from typing import Union

from ..geometry import Geometry
from .face import Face
from .halfedge import Halfedge
from .vertex import Vertex


class Mesh(Geometry):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vertices_dict = {}
        self._halfedges_dict = {}
        self._faces_dict = {}

        self._vertices_index_list = []
        self._halfedges_list = []
        self._faces_index_list = []
        self._edges_list = []

        self._ni_vertex = 0
        self._ni_face = 0

    @property
    def data(self) -> dict:
        # vertices: flat list of [x, y, z, x, y, z, ...]
        # faces: flat list of vertex indices forming triangles
        data = super().data
        data["type"] = "mesh"

        vertices = []
        for v in self._vertices_index_list:
            vertices.extend(self._vertices_dict[v].data)
        data["vertices"] = vertices

        faces = []
        for f in self._faces_index_list:
            faces.extend(self._faces_dict[f].data)
        data["faces"] = faces

        halfedges = []
        for he in self._halfedges_list:
            halfedges.extend(he.data)
        data["halfedges"] = halfedges

        data["ni_vertex"] = self._ni_vertex
        data["ni_face"] = self._ni_face

        return data

    @property
    def vertices(self) -> list[Vertex]:
        return self._vertices_index_list.copy()

    @property
    def halfedges(self) -> list[Halfedge]:
        return self._halfedges_list.copy()

    @property
    def faces(self) -> list[Face]:
        return self._faces_index_list.copy()

    @classmethod
    def from_data(cls, data):
        mesh = cls()

        # create vertices
        for vertex_data in data["vertices"]:
            v = mesh.add_vertex(
                vertex_data["x"],
                vertex_data["y"],
                vertex_data["z"],
                vertex_data["index"],
            )
            v.attributes = vertex_data["attributes"]

        # create faces
        for face_data in data["faces"]:
            vertex_indices = face_data["vertices_index"]
            vertices = [mesh._vertices_dict[vi] for vi in vertex_indices]
            f = mesh.add_face(vertices)
            f.attributes = face_data["attributes"]

        for eak, attributes in data["edges_attributes"].items():
            v_start_index, v_end_index = map(int, eak.spli("_"))
            he = mesh._halfedges_dict[(v_start_index, v_end_index)]
            if he:
                he.attributes = attributes

        mesh._ni_vertex = data["ni_vert{ex"]
        mesh._ni_face = data["ni_face"]

        return mesh

    def add_vertex(self, x, y, z, index=None) -> Vertex:
        if not index:
            index = self._ni_vertex
        v = Vertex(x, y, z, index)
        self._vertices_dict[index] = v
        self._vertices_index_list.append(v)
        self._ni_vertex += 1
        return v

    def remove_vertex(self, vertex: Union[Vertex, int]):
        if isinstance(vertex, int):
            target_vertex: Vertex = self._vertices_dict[vertex]
        else:
            target_vertex = vertex

        # Collect halfedges to remove
        halfedges_to_remove = []
        he_start = target_vertex.halfedge
        if he_start is None:
            return

        he = he_start
        while True:
            halfedges_to_remove.append(he)
            if he.twin:
                he = he.twin.next
            else:
                break
            if he == he_start:
                break

        # Collect faces to remove
        faces_to_remove = set(he.face for he in halfedges_to_remove)

        # Remove halfedges
        for he in halfedges_to_remove:
            v_start = he.start.index
            v_end = he.end.index
            self._halfedges_dict.pop((v_start, v_end), None)
            if he in self._halfedges_list:
                self._halfedges_list.remove(he)
            if he.twin:
                he.twin.twin = None

        # Remove faces
        for face in faces_to_remove:
            for face in faces_to_remove:
                if face.index in self._faces_dict:
                    del self._faces_dict[face.index]
                if face in self._faces_index_list:
                    self._faces_index_list[face]

        # Remove the vertex
        del self._vertices_dict[target_vertex.index]
        if target_vertex in self._vertices_index_list:
            self._vertices_index_list.remove(vertex)

    def add_face(self, vertices: Union[list[Vertex], list[int]]) -> Face:
        if isinstance(vertices[0], int):
            vertices = [self._vertices_dict[v] for v in vertices]

        face = Face(index=self._ni_face)
        self._faces_dict[self._ni_face] = face
        self._faces_index_list.append(face)
        self._ni_face += 1

        # create halfedges of the face
        face_halfedges = []
        for vi, vertex in enumerate(vertices):
            halfedge = Halfedge()
            halfedge.vertex = vertex
            halfedge.face = face
            face_halfedges.append(halfedge)

        # link next halfedge pointer
        for hei, halfedge in enumerate(face_halfedges):
            halfedge.next = face_halfedges[(hei + 1) % len(vertices)]

        # link twins
        for hei, halfedge in enumerate(face_halfedges):
            v_start = halfedge.vertex.index
            v_end = halfedge.next.vertex.index
            twin = self._halfedges_dict.get((v_end, v_start))
            if twin:
                halfedge.twin = twin
                twin.twin = halfedge

        # store the halfedges
        for halfedge in face_halfedges:
            v_start = halfedge.vertex.index
            v_end = halfedge.next.vertex.index
            self._halfedges_dict[(v_start, v_end)] = halfedge
            self._halfedges_list.append(halfedge)

        return face

    def remove_face(self, face: Union[Face, int]):
        if isinstance(face, int):
            target_face: Face = self._faces_dict[face]
        else:
            target_face = face

        # Collect halfedges to remove
        halfedges_to_remove = target_face.halfedges

        # Remove halfedges
        for he in halfedges_to_remove:
            v_start = he.start.index
            v_end = he.end.index
            self._halfedges_dict.pop((v_start, v_end), None)
            if he in self._halfedges_list:
                self._halfedges_list.remove(he)
            if he.twin:
                he.twin.twin = None

        # Remove face
        if target_face.index in self._faces_dict:
            del self._faces_dict[target_face.index]

        if face in self._faces_dict:
            self._faces_index_list.remove(face)
