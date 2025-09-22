import math


from point import Point
from vector import Vector
from frame import Frame



class Cuboid:

    def __init__(self, frame: Frame, lx: float, ly: float, lz: float):
        self.frame = frame
        self.lx = lx
        self.ly = ly
        self.lz = lz


    def __repr__(self):
        return f"Cuboid(center={self.frame.origin}, lx={self.lx}, ly={self.ly}, lz={self.lz})"


    @property
    def volume(self) -> float:
        return self.lx * self.ly * self.lz
    


    @property
    def area(self) -> float:
        """Surface area of the cuboid."""
        return 2 * (self.lx * self.ly + self.ly * self.lz + self.lx * self.lz)
    


    @property
    def vertices(self) -> list[Point]:

        center = self.frame.origin
        x = self.frame.xaxis
        y = self.frame.yaxis
        z = self.frame.zaxis

        hx = self.lx / 2
        hy = self.ly / 2
        hz = self.lz / 2

        v0 = center + (x * -hx) + (y * -hy) + (z * -hz)
        v1 = center + (x * hx) + (y * -hy) + (z * -hz)
        v2 = center + (x * hx) + (y * hy) + (z * -hz)
        v3 = center + (x * -hx) + (y * hy) + (z * -hz)
        v4 = center + (x * -hx) + (y * -hy) + (z * hz)
        v5 = center + (x * hx) + (y * -hy) + (z * hz)
        v6 = center + (x * hx) + (y * hy) + (z * hz)
        v7 = center + (x * -hx) + (y * hy) + (z * hz)

        return [v0, v1, v2, v3, v4, v5, v6, v7]
    

    @property
    def diagonal_length(self) -> float:
        """Length of the main diagonal."""
        return math.sqrt(self.lx**2 + self.ly**2 + self.lz**2)
    



    def contains(self, point: Point) -> bool:
        """Check if point lies inside the cuboid (in local frame coords)."""
        # Express point relative to cuboid frame
        v = Vector(point.x - self.frame.origin.x,
                   point.y - self.frame.origin.y,
                   point.z - self.frame.origin.z)

        # Project onto axes
        dx = v.dot(self.frame.xaxis)
        dy = v.dot(self.frame.yaxis)
        dz = v.dot(self.frame.zaxis)

        return (abs(dx) <= self.lx/2 and
                abs(dy) <= self.ly/2 and
                abs(dz) <= self.lz/2)

