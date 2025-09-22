import math

from vector import Vector

class Point:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_vector(self):
        """Return a Vector from the origin to this point."""
        return Vector(self.x, self.y, self.z)

    def __sub__(self, other):
        """Subtract another point -> returns a Vector."""
        if isinstance(other, Point):
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        raise TypeError("Subtraction only supported between Points")

    def __add__(self, vec):
        """Add a Vector -> returns a new Point."""
        if isinstance(vec, Vector):
            return Point(self.x + vec.x, self.y + vec.y, self.z + vec.z)
        raise TypeError("Can only add a Vector to a Point")

    def distance_to(self, other):
        """Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def __repr__(self):
        return f"Point({self.x}, {self.y}, {self.z})"
