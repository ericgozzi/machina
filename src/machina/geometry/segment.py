from point import Point
from vector import Vector
from line import Line


class Segment:

    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Segment(start={self.start}, end={self.end})"
    

    @property
    def direction(self) -> Vector:
        return(self.end - self.start).normalized()
    
    @property
    def length(self):
        return (self.end - self.start).magnitude
    
    @property
    def line(self) -> Line:
        return Line(self.start, self.direction)
