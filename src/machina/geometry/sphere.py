import math

from .point import Point


class Sphere:

    def __init__(self, center: Point, radius: float):
        self.center = center
        self.radius = radius


    def __repr__(self):
        return f"Sphere(center={self.center}, radius={self.radius})"


    @property
    def area(self):
        return 4 * math.pi * self.radius**2
    

    @property
    def volume(self):
        return (4/3) * math.py * self.radius ** 3
    

    def contains(self, point: Point) -> bool:
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z
        return dx * dx + dy * dy + dz * dz <= self.radius**2
    
    