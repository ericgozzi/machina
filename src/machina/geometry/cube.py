from cuboid import Cuboid
from frame import Frame



class Cube(Cuboid):

    def __init__(self, frame: Frame, size: float):
        super().__init__(frame, size, size, size)

    def __repr__(self):
        return f"Cube(center={self.frame.origin}, size={self.lx})"
