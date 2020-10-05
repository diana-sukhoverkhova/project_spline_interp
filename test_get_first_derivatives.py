import numpy as np
from numpy.testing import assert_allclose

import pytest

from cyclic_interpolate import (CyclicInterpCurve,
        get_first_derivatives)


def test_none_x():
    x = None
    y = np.arange(8)
    with pytest.raises(ValueError):
        get_first_derivatives(x, y)

def test_none_y():
    x = np.arange(8)
    y = None
    with pytest.raises(ValueError):
        get_first_derivatives(x, y)

def test_none():
    x = None
    y = None
    with pytest.raises(ValueError):
        get_first_derivatives(x, y)

def test_non_sort():
    rndm = np.random.RandomState(1234)
    x = np.sort(rndm.uniform(size=8))
    y = np.random.uniform(size=8)
    with pytest.raises(ValueError):
        get_first_derivatives(x, y)

def test_num_points():
    x = np.arange(2)
    y = np.arange(2)

    with pytest.raises(ValueError):
        get_first_derivatives(x, y)

def test_non_periodic():
    x = np.arange(8)
    y = np.arange(8)

    with pytest.raises(ValueError):
        get_first_derivatives(x, y)

def test_shapes_x_greater():
    rndm = np.random.RandomState(1234)
    x = np.sort(rndm.uniform(size=9))
    y = np.random.uniform(size=8)
    y[-1] = y[0]

    with pytest.raises(ValueError):
        get_first_derivatives(x, y)

def test_shapes_x_less():
    rndm = np.random.RandomState(1234)
    x = np.sort(rndm.uniform(size=8))
    y = np.random.uniform(size=9)
    y[-1] = y[0]

    with pytest.raises(ValueError):
        get_first_derivatives(x, y)

def test_x_asc():
    x = np.array([1,2,3,2.5,1.5])
    y = np.arange(5)
    with pytest.raises(ValueError):
        get_first_derivatives(x, y)

def test_matrix():
    rndm = np.random.RandomState(1234)
    x = np.sort(rndm.uniform(size=8))
    y = np.random.uniform(size=8)
    y[-1] = y[0]

    der = get_first_derivatives(x, y)
    assert_allclose(der.shape[0], x.shape[0], atol=1e-12)
