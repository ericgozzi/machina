from point import Point
from vector import Vector


class Frame:

    def __init__(self, origin: Point, xaxis=None, yaxis=None):
        self.origin = origin
        self.xaxis = xaxis.normalized() if xaxis else Vector(1, 0, 0)
        self.yaxis = yaxis.normalized() if yaxis else Vector(0, 1, 0)


    def __repr__(self):
        return f"Frame(origin={self.origin}, xaxis={self.xaxis}, yaxis={self.yaxis}, zaxis={self.zaxis})"


    @property
    def zaxis(self):
        return self.xaxis.cross(self.yaxis).normalized()
    

    @classmethod
    def from_normal(cls, origin: Point, normal: Vector):
        zaxis = normal.normalized()

        # Pick a helper vector that is not parallel to zaxis
        if abs(zaxis.x) < 0.9:
            helper = Vector(1, 0, 0)
        else:
            helper = Vector(0, 1, 0)

        # Build orthogonal x- and y-axes
        xaxis = helper.cross(zaxis).normalized()
        yaxis = zaxis.cross(xaxis).normalized()

        return cls(origin, xaxis, yaxis)
