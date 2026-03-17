from machina.geometry import Point, Vector


def test_point_creation():
    """
    Tests the creation of a Point object.
    """
    p = Point(1, 2, 3)
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3


def test_point_data():
    """
    Tests the data property of the Point class.
    """
    p = Point(1, 2, 3)
    data = p.data
    assert data["x"] == 1
    assert data["y"] == 2
    assert data["z"] == 3

    reconstructed = Point.from_data(data)
    assert reconstructed == p


def test_point_addition():
    """
    Tests the addition of a Point and a Vector.
    """
    p = Point(1, 2, 3)
    v = Vector(4, 5, 6)
    p2 = p + v
    assert isinstance(p2, Point)
    assert p2.x == 5
    assert p2.y == 7
    assert p2.z == 9


def test_point_subtraction():
    """
    Tests the subtraction of two points.
    """
    p1 = Point(1, 2, 3)
    p2 = Point(4, 5, 6)
    v = p2 - p1
    assert isinstance(v, Vector)
    assert v.x == 3
    assert v.y == 3
    assert v.z == 3


def test_point_equality():
    """
    Tests the equality of two points.
    """
    p1 = Point(1, 2, 3)
    p2 = Point(1, 2, 3)
    p3 = Point(4, 5, 6)
    assert p1 == p2
    assert p1 != p3
