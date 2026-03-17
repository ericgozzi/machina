from machina import Frame, Point, Vector


def test_frame_creation():
    """
    Tests the creation of a Frame object.
    """
    p = Point(1, 2, 3)
    x = Vector(1, 0, 0)
    y = Vector(0, 1, 0)
    frame = Frame(p, x, y)
    assert frame.origin == p
    assert frame.x_axis == x
    assert frame.y_axis == y


def test_frame_data():
    """
    Tests the data property of the Frame class.
    """
    p = Point(1, 2, 3)
    x = Vector(1, 0, 0)
    y = Vector(0, 1, 0)
    frame = Frame(p, x, y)
    data = frame.data
    assert data["type"] == "frame"
    assert data["origin"]["x"] == 1
    assert data["origin"]["y"] == 2
    assert data["origin"]["z"] == 3
    assert data["x_axis"]["x"] == 1
    assert data["x_axis"]["y"] == 0
    assert data["x_axis"]["z"] == 0
    assert data["y_axis"]["x"] == 0
    assert data["y_axis"]["y"] == 1
    assert data["y_axis"]["z"] == 0

    reconstructed = Frame.from_data(data)
    assert reconstructed.origin == frame.origin
    assert reconstructed.x_axis == frame.x_axis
    assert reconstructed.y_axis == frame.y_axis


def test_frame_z_axis():
    """
    Tests the z_axis property of the Frame class.
    """
    p = Point(1, 2, 3)
    x = Vector(1, 0, 0)
    y = Vector(0, 1, 0)
    frame = Frame(p, x, y)
    z = frame.z_axis
    assert z.x == 0
    assert z.y == 0
    assert z.z == 1


def test_frame_origin():
    """
    Tests the origin property of the Frame class.
    """
    p = Point(1, 2, 3)
    x = Vector(1, 0, 0)
    y = Vector(0, 1, 0)
    frame = Frame(p, x, y)
    origins = frame.origin
    assert origins.x == 1
    assert origins.y == 2
    assert origins.z == 3
