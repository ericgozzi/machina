from ..frame import Frame
from ..point import Point
from ..vector import Vector


class Transformation:
    def __init__(self, matrix: list[list[float]]):
        self.matrix = matrix

    @property
    def data(self) -> dict:
        return {"type": "transformation", "matrix": self.matrix}

    @property
    def scale_x(self) -> float:
        # Length of the first column (X-basis vector)
        return (
            self.matrix[0][0] ** 2 + self.matrix[1][0] ** 2 + self.matrix[2][0] ** 2
        ) ** 0.5

    @property
    def scale_y(self) -> float:
        # Length of the second column (Y-basis vector)
        return (
            self.matrix[0][1] ** 2 + self.matrix[1][1] ** 2 + self.matrix[2][1] ** 2
        ) ** 0.5

    @property
    def scale_z(self) -> float:
        # Length of the third column (Z-basis vector)
        return (
            self.matrix[0][2] ** 2 + self.matrix[1][2] ** 2 + self.matrix[2][2] ** 2
        ) ** 0.5

    @property
    def scale_factors(self):
        """Returns (sx, sy, sz) as a tuple"""
        return (self.scale_x, self.scale_y, self.scale_z)

    @classmethod
    def from_data(cls, data: dict):
        return cls(data["matrix"])

    @classmethod
    def identity(cls):
        matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return cls(matrix)

    def as_frame(self) -> Frame:

        frame_origin = Point(self.matrix[0][3], self.matrix[1][3], self.matrix[2][3])

        sx, sy, sz = self.scale_factors
        xaxis = Vector(
            self.matrix[0][0] / sx, self.matrix[1][0] / sx, self.matrix[2][0] / sx
        )
        yaxis = Vector(
            self.matrix[0][1] / sy, self.matrix[1][1] / sy, self.matrix[2][1] / sy
        )

        return Frame(frame_origin, xaxis, yaxis)

    def apply_to_point(self, point: Point) -> Point:
        # Uses 1 as the 4th component (Translation affects it)
        px, py, pz = point.x, point.y, point.z

        nx = (
            self.matrix[0][0] * px
            + self.matrix[0][1] * py
            + self.matrix[0][2] * pz
            + self.matrix[0][3]
        )
        ny = (
            self.matrix[1][0] * px
            + self.matrix[1][1] * py
            + self.matrix[1][2] * pz
            + self.matrix[1][3]
        )
        nz = (
            self.matrix[2][0] * px
            + self.matrix[2][1] * py
            + self.matrix[2][2] * pz
            + self.matrix[2][3]
        )

        return Point(nx, ny, nz)

    def apply_to_vector(self, vector):
        # Uses 0 as the 4th component (Translation is IGNORED)
        vx, vy, vz = vector.x, vector.y, vector.z

        nx = self.matrix[0][0] * vx + self.matrix[0][1] * vy + self.matrix[0][2] * vz
        ny = self.matrix[1][0] * vx + self.matrix[1][1] * vy + self.matrix[1][2] * vz
        nz = self.matrix[2][0] * vx + self.matrix[2][1] * vy + self.matrix[2][2] * vz

        return Vector(nx, ny, nz)
