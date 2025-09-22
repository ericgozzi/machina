import math
from point import Point




class Circle:

    def __init__(self, center: Point, radius: float):
        self.center = center
        self.radius = radius


    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"
    
    
    @property
    def area(self):
        return math.pi * self.radius**2
    
    @property
    def diameter(self):
        return self.radius * 2
    
    @property
    def perimeter(self):
        return 2 * math.pi * self.radius