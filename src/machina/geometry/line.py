from point import Point
from vector import Vector



class Line:

    def __init__(self, point: Point, direction: Vector):
        self.point = point
        self.direction = direction.normalized()



    def __repr__(self):
        return f"Line(point={self.point}, direction={self.direction})"



    def point_at(self, t):
        """Return a point on the line at parameter t."""
        return self.point + self.direction * t
    
    
    def closest_point(self, point: Point):
        """Return the point on the line closest to another `Point`."""
        v: Vector = point - self.point
        t = v.dot(self.direction)
        return self.point_at(t)