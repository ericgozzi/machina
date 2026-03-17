from machina.geometry import Line, Point


def test_line_creation():
    """
    Tests the creation of a Line object.
    """
    p1 = Point(1, 2, 3)
    p2 = Point(4, 5, 6)
    line = Line(p1, p2)
    assert line.start == p1
    assert line.end == p2


def test_line_data():
    """
    Tests the data property of the Line class.
    """
    p1 = Point(1, 2, 3)
    p2 = Point(4, 5, 6)
    line = Line(p1, p2)
    data = line.data
    assert data["type"] == "line"
    assert data["start"]["x"] == 1
    assert data["start"]["y"] == 2
    assert data["start"]["z"] == 3
    assert data["end"]["x"] == 4
    assert data["end"]["y"] == 5
    assert data["end"]["z"] == 6

    reconstructed = Line.from_data(data)
    assert reconstructed.start == line.start
    assert reconstructed.end == line.end


def test_line_length():
    """
    Tests the length property of the Line class.
    """
    p1 = Point(0, 0, 0)
    p2 = Point(3, 4, 0)
    line = Line(p1, p2)
    assert line.length == 5.0

    p3 = Point(1, 1, 1)
    p4 = Point(1, 1, 1)
    line2 = Line(p3, p4)
    assert line2.length == 0.0


def test_line_midpoint():
    """
    Tests the midpoint property of the Line class.
    """
    p1 = Point(0, 0, 0)
    p2 = Point(10, 20, 30)
    line = Line(p1, p2)
    midpoint = line.midpoint
    assert midpoint.x == 5
    assert midpoint.y == 10
    assert midpoint.z == 15
