from vertex import Vertex
from face import Face


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



