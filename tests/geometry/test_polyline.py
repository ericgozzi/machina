from machina import Point, Polyline


def test_polyline_init():
    p1 = Point(0, 0, 0)
    p2 = Point(1, 1, 1)
    p3 = Point(2, 2, 2)
    polyline = Polyline([p1, p2, p3])
    assert len(polyline.points) == 3
    assert polyline.points[0] == p1
    assert polyline.points[1] == p2
    assert polyline.points[2] == p3


def test_polyline_data():
    p1 = Point(0, 0, 0)
    p2 = Point(1, 1, 1)
    polyline = Polyline([p1, p2])
    data = polyline.data
    assert data["type"] == "polyline"
    assert len(data["points"]) == 2
    assert data["points"][0] == p1.data
    assert data["points"][1] == p2.data


def test_polyline_is_closed():
    # Open polyline
    p1 = Point(0, 0, 0)
    p2 = Point(1, 1, 1)
    p3 = Point(2, 2, 2)
    open_polyline = Polyline([p1, p2, p3])
    assert not open_polyline.is_closed

    # Closed polyline
    p4 = Point(0, 0, 0)
    closed_polyline = Polyline([p1, p2, p3, p4])
    assert closed_polyline.is_closed


def test_polyline_from_data():
    p1 = Point(0, 0, 0)
    p2 = Point(1, 1, 1)
    p3 = Point(2, 2, 2)
    polyline = Polyline([p1, p2, p3])
    data = polyline.data
    reconstructed = Polyline.from_data(data)
    assert isinstance(reconstructed, Polyline)
    assert len(reconstructed.points) == 3
    assert reconstructed.points[0] == polyline.points[0]
    assert reconstructed.points[1] == polyline.points[1]
    assert reconstructed.points[2] == polyline.points[2]


def test_polyline_close():
    # Close an open polyline
    p1 = Point(0, 0, 0)
    p2 = Point(1, 1, 1)
    p3 = Point(2, 2, 2)
    polyline = Polyline([p1, p2, p3])
    assert not polyline.is_closed

    closed_polyline = polyline.close()
    assert closed_polyline.is_closed
    assert len(closed_polyline.points) == 4
    assert closed_polyline.points[-1] == p1
    assert id(polyline) == id(closed_polyline)  # check it returns self

    # "Close" an already closed polyline
    p4 = Point(0, 0, 0)
    already_closed = Polyline([p1, p2, p3, p4])
    assert already_closed.is_closed

    result = already_closed.close()
    assert result.is_closed
    assert len(result.points) == 4  # should not add another point
    assert id(already_closed) == id(result)
