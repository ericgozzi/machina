from __future__ import annotations


class Color:
    def __init__(self, r: float, g: float, b: float):
        self.r = r
        self.g = g
        self.b = b

    def __str__(self):
        return f"Color({self.r}, {self.g}, {self.b})"

    @property
    def data(self) -> dict:
        return {"r": self.r, "g": self.g, "b": self.b}

    @property
    def r(self) -> float:
        return self._r

    @r.setter
    def r(self, value: float):
        if not (0 <= value <= 1):
            raise ValueError("Red value must be between 0 and 1")
        self._r = value

    @property
    def g(self) -> float:
        return self._g

    @g.setter
    def g(self, value: float):
        if not (0 <= value <= 1):
            raise ValueError("Green value must be between 0 and 1")
        self._g = value

    @property
    def b(self) -> float:
        return self._b

    @b.setter
    def b(self, value: float):
        if not (0 <= value <= 1):
            raise ValueError("Blue value must be between 0 and 1")
        self._b = value

    @classmethod
    def from_data(cls, data: dict) -> Color:
        return cls(data["r"], data["g"], data["b"])

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int) -> Color:
        return cls(r / 255.0, g / 255.0, b / 255.0)
