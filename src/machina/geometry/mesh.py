from machina.geometry.face import Face
from machina.geometry.geometry import Geometry
from machina.geometry.halfedge import Halfedge
from machina.geometry.vertex import Vertex


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

    def add_vertex(self, x, y, z, index=None):
        if not index:
            index = self._ni_vertex
        v = Vertex(x, y, z, index)
        self._vertices_dict[index] = v
        self._vertices_index_list.append(v)
        self._ni_vertex += 1

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
