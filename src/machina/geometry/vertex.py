from point import Point
from edge import HalfEdge


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