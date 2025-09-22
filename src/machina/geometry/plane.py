from point import Point
from vector import Vector


class Plane:

    def __init__(self, point: Point, vector_a: Vector, vector_b: Vector):
        self.point = point
        self.vector_a = vector_a
        self.vector_b = vector_b


    def __repr__(self):
        return f"Plane(point:{self.point}, vector_a:{self.vector_a}, vector_b:{self.vector_b})"

    @property
    def normal(self) -> Vector:
        """Return the normal vector of the plane (v_a × v_b)"""
        return self.vector_a.cross(self.vector_b)


    def get_point(self, s: float, t: float) -> Point:
        """Return a point on the plane: P = P0 + s*v_a + t*v_b"""
        return self.point + self.vector_a * s + self.vector_b * t
