from .cuboid import Cuboid
from .frame import Frame
from .point import Point



class Vertex:

    def __init__(self, point: Point):
        """
        A vertex in the mesh.

        :param point: Coordinates of the vertex (could be a Point object or a tuple/list of 3 floats)
        """ 
        self.point = point
        # one outgoing half-edge from this vertex
        self.halfedge: HalfEdge = None

        self.attributes = {}
        

    def add_attribute(self, key: any, value: any):
        self.attributes[key] = value

    def get_attribute(self, key: any) -> any:
        return self.attributes[key]




class Face:

    def __init__(self, vertex_indices: list[int]):
        """
        A face in the mesh.

        :param vertex_indices: List of indices into the mesh's vertex list
        """
        # store the face vertices
        self.vertex_indices = vertex_indices
        # one halfedge bordering the face
        self.halfedge = None



class HalfEdge:

    def __init__(self):
        """
        A half-edge in the mesh.
        """

        # vertex at the end of this half-edge
        self.vertex: Vertex = None

        # opposite half-edge
        self.twin: HalfEdge = None

        # next half.edge around the face
        self.next: HalfEdge = None

        # previous half-edge around the face
        self.prev: HalfEdge = None

        # face this half-edge borders
        self.face: Face = None

        # optional reference to edge object







class Mesh:
    def __init__(self, vertices: list[Vertex] = None, faces: list[Face] = None):
        """
        Construct a half-edge mesh.

        :param vertices: List of Vertex objects
        :param faces: List of Face objects, each containing vertex indices
        """
        self.vertices = vertices if vertices else []
        self.faces = faces if faces else []
        self.halfedges = []       # List to store all half-edges
        self.edge_map = {}        # Map from (start_vertex, end_vertex) -> HalfEdge for twins

        if self.faces:
            self.build_halfedge_mesh()





    def build_halfedge_mesh(self):
        """
        Build half-edge connectivity for all faces in the mesh.
        """
        for face in self.faces:
            face_halfedges = []          # Half-edges for this face
            n = len(face.vertex_indices) # Number of vertices in this face

            # Step 1: create half-edges for the face
            for i in range(n):
                he = HalfEdge()
                he.face = face
                # The half-edge points to the next vertex in the loop
                he.vertex = self.vertices[face.vertex_indices[(i + 1) % n]]
                face_halfedges.append(he)
                self.halfedges.append(he)

            # Step 2: link next and previous half-edges
            for i in range(n):
                he = face_halfedges[i]
                he.next = face_halfedges[(i + 1) % n]
                he.prev = face_halfedges[(i - 1) % n]

            # Step 3: assign one half-edge to the face
            face.halfedge = face_halfedges[0]

            # Step 4: assign an outgoing half-edge to each vertex if not set
            for he in face_halfedges:
                if he.prev.vertex.halfedge is None:
                    he.prev.vertex.halfedge = he

            # Step 5: link twin half-edges
            for he in face_halfedges:
                start = he.prev.vertex
                end = he.vertex
                key = (start, end)      # Use vertex objects directly as keys
                twin_key = (end, start) # Opposite direction

                if twin_key in self.edge_map:
                    twin = self.edge_map[twin_key]
                    he.twin = twin
                    twin.twin = he

                # Store this half-edge for potential twins in future faces
                self.edge_map[key] = he

    def __repr__(self):
        return f"<Mesh: {len(self.vertices)} vertices, {len(self.halfedges)} halfedges, {len(self.faces)} faces>"
    



    @classmethod
    def from_cuboid(cls, cuboid: Cuboid):

        cuboid_points = cuboid.vertices

        f1 = Face([0, 3, 2, 1])
        f2 = Face([0, 4, 7, 3])
        f3 = Face([0, 1, 5, 4])
        f4 = Face([1, 2, 6, 5])
        f5 = Face([2, 3, 7, 6])
        f6 = Face([4, 5, 6, 7])

        cuboid_vertices = [Vertex(point) for point in cuboid_points]

        mesh_cuboid = cls(cuboid_vertices, [f1, f2, f3, f4, f5, f6])

        return mesh_cuboid
    







    