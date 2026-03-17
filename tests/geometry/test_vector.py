import math

from machina.geometry import Vector


def test_vector_creation():
    """
    Tests the creation of a Vector object.
    """
    v = Vector(1, 2, 3)
    assert v.x == 1
    assert v.y == 2
    assert v.z == 3


def test_vector_data():
    """
    Tests the data property of the Vector class.
    """
    v = Vector(1, 2, 3)
    data = v.data
    assert data["type"] == "vector"
    assert data["x"] == 1
    assert data["y"] == 2
    assert data["z"] == 3

    reconstructed = Vector.from_data(data)
    assert reconstructed == v


def test_vector_length():
    """
    Tests the length property of the Vector class.
    """
    v = Vector(3, 4, 0)
    assert v.length == 5.0

    v2 = Vector(0, 0, 0)
    assert v2.length == 0.0


def test_vector_addition():
    """
    Tests the addition of two vectors.
    """
    v1 = Vector(1, 2, 3)
    v2 = Vector(4, 5, 6)
    v3 = v1 + v2
    assert isinstance(v3, Vector)
    assert v3.x == 5
    assert v3.y == 7
    assert v3.z == 9


def test_vector_subtraction():
    """
    Tests the subtraction of two vectors.
    """
    v1 = Vector(1, 2, 3)
    v2 = Vector(4, 5, 6)
    v3 = v2 - v1
    assert isinstance(v3, Vector)
    assert v3.x == 3
    assert v3.y == 3
    assert v3.z == 3


def test_vector_multiplication():
    """
    Tests the multiplication of a vector by a scalar.
    """
    v1 = Vector(1, 2, 3)
    v2 = v1 * 2
    assert isinstance(v2, Vector)
    assert v2.x == 2
    assert v2.y == 4
    assert v2.z == 6


def test_vector_division():
    """
    Tests the division of a vector by a scalar.
    """
    v1 = Vector(2, 4, 6)
    v2 = v1 / 2
    assert isinstance(v2, Vector)
    assert v2.x == 1
    assert v2.y == 2
    assert v2.z == 3


def test_vector_equality():
    """
    Tests the equality of two vectors.
    """
    v1 = Vector(1, 2, 3)
    v2 = Vector(1, 2, 3)
    assert v1 == v2
    v3 = Vector(4, 5, 6)
    assert v1 != v3


def test_vector_cross_product():
    """
    Tests the cross product of two vectors.
    """
    v1 = Vector(1, 0, 0)
    v2 = Vector(0, 1, 0)
    v3 = v1.cross(v2)
    assert isinstance(v3, Vector)
    assert v3.x == 0
    assert v3.y == 0
    assert v3.z == 1


def test_vector_dot_product():
    """
    Tests the dot product of two vectors.
    """
    v1 = Vector(1, 2, 3)
    v2 = Vector(4, 5, 6)
    dot = v1.dot(v2)
    assert dot == 32


def test_vector_unitize():
    """
    Tests the unitize method of the Vector class.
    """
    v = Vector(3, 4, 0)
    v.unitize()
    assert math.isclose(v.length, 1.0)
    assert math.isclose(v.x, 0.6)
    assert math.isclose(v.y, 0.8)
    assert math.isclose(v.z, 0.0)
