from machina.geometry import Cuboid, Frame, Point, Vector


def test_cuboid_creation():
    cuboid = Cuboid(1.0, 2.0, 3.0, None)
    world_frame = Frame.world()
    assert cuboid.x_size == 1.0
    assert cuboid.y_size == 2.0
    assert cuboid.z_size == 3.0
    assert cuboid.frame.origin == world_frame.origin
    assert cuboid.frame.x_axis == world_frame.x_axis
    assert cuboid.frame.y_axis == world_frame.y_axis
    assert cuboid.frame.z_axis == world_frame.z_axis


def test_cuboid_creation_2():
    frame = Frame(Point(1, 2, 3), Vector(0, 1, 0), Vector(0, 0, 1))
    cuboid = Cuboid(1.0, 2.0, 3.0, frame)
    assert cuboid.x_size == 1.0
    assert cuboid.y_size == 2.0
    assert cuboid.z_size == 3.0
    assert cuboid.frame.origin == Point(1, 2, 3)
    assert cuboid.frame.x_axis == Vector(0, 1, 0)
    assert cuboid.frame.y_axis == Vector(0, 0, 1)
    assert cuboid.frame.z_axis == Vector(1, 0, 0)


def test_cuboid_data():
    cuboid = Cuboid(1.0, 2.0, 3.0, None)
    data = cuboid.data
    assert data["x_size"] == 1.0
    assert data["y_size"] == 2.0
    assert data["z_size"] == 3.0
    assert data["frame"]["origin"] == Point(0, 0, 0).data
    assert data["frame"]["x_axis"] == Vector(1, 0, 0).data
    assert data["frame"]["y_axis"] == Vector(0, 1, 0).data

    reconstructed = Cuboid.from_data(data)
    assert reconstructed.x_size == 1.0
    assert reconstructed.y_size == 2.0
    assert reconstructed.z_size == 3.0
    assert reconstructed.frame.origin == Point(0, 0, 0)
    assert reconstructed.frame.x_axis == Vector(1, 0, 0)
    assert reconstructed.frame.y_axis == Vector(0, 1, 0)
    assert reconstructed.frame.z_axis == Vector(0, 0, 1)


def test_cuboid_volume():
    cuboid = Cuboid(1.0, 2.0, 3.0, None)
    assert cuboid.volume == 6.0


def test_cuboid_vertices():
    cuboid = Cuboid(2.0, 2.0, 2.0, None)
    v0, v1, v2, v3, v4, v5, v6, v7 = cuboid.vertices
    print(cuboid.vertices)
    assert v0 == Point(-1, -1, -1)
    assert v1 == Point(1, -1, -1)
    assert v2 == Point(1, 1, -1)
    assert v3 == Point(-1, 1, -1)
    assert v4 == Point(-1, -1, 1)
    assert v5 == Point(1, -1, 1)
    assert v6 == Point(1, 1, 1)
    assert v7 == Point(-1, 1, 1)
