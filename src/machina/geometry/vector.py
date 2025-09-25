import math

class Vector:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


    def __repr__(self):
            return f"Vector({self.x}, {self.y}, {self.z})"



    @property
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    

    # Vector addition
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
     # Vector subtraction
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
    
    # Scalar multiplication
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
    # support scalar * vector
    __rmul__ = __mul__ 
    
    # Scalar division
    def __truediv__(self, scalar):
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)
    




    def normalize(self):
        """Normalize the vector in place."""
        m = self.magnitude
        if m != 0:
            self.x /= m
            self.y /= m
            self.z /= m
        return self  # allow chaining
    

    def normalized(self):
        """Return a new normalized vector."""
        m = self.magnitude
        if m == 0:
            return Vector(0, 0, 0)
        return Vector(self.x / m, self.y / m, self.z / m)
    




    # Dot product
    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z
    



    # Cross product
    def cross(self, other):
        return Vector(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )

