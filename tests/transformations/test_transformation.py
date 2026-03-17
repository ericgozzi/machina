import math

import pytest

from machina.geometry import Point, Transformation, Vector


def test_transformation_identity():
    """
    Tests the creation of an identity transformation.
    """
    t = Transformation.identity()
    assert t.matrix == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_transformation_from_data():
    """
    Tests creating a transformation from data.
    """
    data = {
        "matrix": [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 3.0, 0.0, 2.0],
            [0.0, 0.0, 4.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
    t = Transformation.from_data(data)
    assert t.matrix == data["matrix"]


def test_transformation_apply_to_point():
    """
    Tests applying a transformation to a point.
    """
    # Translation
    t = Transformation(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    p = Point(1, 2, 3)
    p2 = t.apply_to_point(p)
    assert p2.x == 11
    assert p2.y == 22
    assert p2.z == 33

    # Scale
    t_scale = Transformation(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    p3 = t_scale.apply_to_point(p)
    assert p3.x == 2
    assert p3.y == 6
    assert p3.z == 12


def test_transformation_apply_to_vector():
    """
    Tests applying a transformation to a vector.
    """
    # Translation should be ignored for vectors
    t = Transformation(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    v = Vector(1, 2, 3)
    v2 = t.apply_to_vector(v)
    assert v2.x == 1
    assert v2.y == 2
    assert v2.z == 3

    # Scale
    t_scale = Transformation(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    v3 = t_scale.apply_to_vector(v)
    assert v3.x == 2
    assert v3.y == 6
    assert v3.z == 12


def test_transformation_scale_factors():
    """
    Tests the scale_factors property.
    """
    t = Transformation(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    sx, sy, sz = t.scale_factors
    assert sx == 2.0
    assert sy == 3.0
    assert sz == 4.0

    t_identity = Transformation.identity()
    sx, sy, sz = t_identity.scale_factors
    assert sx == 1.0
    assert sy == 1.0
    assert sz == 1.0
