from machina.geometry import Point, Sphere


def test_sphere_creation():
    sphere = Sphere(Point(2, 3, 4), 10)
    assert sphere.center == Point(2, 3, 4)
    assert sphere.radius == 10


def test_sphere_data():
    sphere = Sphere(Point(2, 3, 4), 10)
    data = sphere.data
    assert data["type"] == "sphere"
    assert data["center"] == Point(2, 3, 4).data
    assert data["radius"] == 10

    reconstructed = Sphere.from_data(data)
    assert reconstructed.center == Point(2, 3, 4)
    assert reconstructed.radius == 10


def test_sphere_volume():
    sphere = Sphere(Point(2, 3, 4), 10)
    assert sphere.volume == 4 / 3 * 3.141592653589793 * 10**3
