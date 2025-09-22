import math


from point import Point
from vector import Vector




class Polyline:

    def __init__(self, points: list[Point]):

        self.points = points

        if len(self.points) < 2:
            raise ValueError("Polyline must have at least 2 points.")
        


    def __repr__(self):
        return f"Polyline({self.points})"
    

    @property
    def length(self) -> float:
        """Total length of the polyline."""
        total = 0
        for i in range(len(self.points) - 1):
            p0 = self.points[i]
            p1 = self.points[i + 1]
            dx = p1.x - p0.x
            dy = p1.y - p0.y
            dz = p1.z - p0.z
            total += math.sqrt(dx*dx + dy*dy + dz*dz)
        return total
  
    
    @property
    def is_closed(self) -> bool:
        """Return True if the polyline is closed."""
        return self.points[0] == self.points[-1]
    

    def add_point(self, point: Point):
        """Add a point to the end of the polyline."""
        self.points.append(point)

    
    def remove_point(self, index: int):
        """Remove a point by index."""
        if 0 <= index < len(self.points):
            self.points.pop(index)
        else:
            raise IndexError("Point index out of range.")
        

    def close(self):
        """Close the polyline by adding the first point at the end if not already closed."""
        if not self.is_closed:
            self.points.append(self.points[0])