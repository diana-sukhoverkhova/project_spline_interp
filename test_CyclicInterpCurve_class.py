import numpy as np
from numpy.testing import assert_allclose
import pytest

from cyclic_interpolate import (CyclicInterpCurve, make_spline)


def test_init_x():
    x = None
    y = np.arange(8)
    with pytest.raises(ValueError):
        make_spline(x, y)

def test_init_y():
    x = np.arange(8)
    y = None
    with pytest.raises(ValueError):
        make_spline(x, y)

def test_init():
    x = None
    y = None
    with pytest.raises(ValueError):
        make_spline(x, y)

def test_call_less():
    x = np.array([5, 6, 7, 8])
    y = np.array([1, 2, 3, 1])
    der = np.array([4, 5, 6, 7])
    test = CyclicInterpCurve(x, y, der)
    with pytest.raises(ValueError):
        test.__call__(4)

def test_call_great():
    x = np.array([5, 6, 7, 8])
    y = np.array([1,2,3,1])
    der = np.array([4, 5, 6, 7])
    test = CyclicInterpCurve(x, y, der)
    with pytest.raises(ValueError):
        test.__call__(9)

def test_strict_asc():
    x = np.array([5, 6, 6, 8])
    y = np.array([1, 2, 3, 1])
    der = np.array([4, 5, 6, 7])
    with pytest.raises(ValueError):
        CyclicInterpCurve(x, y, der)

def test_random():
    rndm = np.random.RandomState(1234)
    x = np.sort(rndm.uniform(size=8))
    y = np.random.uniform(size=8)
    y[-1] = y[0]

    spl = make_spline(x, y)
    ynew = [spl(_) for _ in x]
    assert_allclose(ynew, y, atol=1e-12)
