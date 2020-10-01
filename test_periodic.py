import numpy as np
from numpy.testing import assert_allclose

import pytest

from cyclic_interpolate import (cyclic_interp_curve,
        cubic_spline_interpolation_first_derivatives)


def make_spline(x, y):
    # A convenience wrapper for the ctor.
    # FIXME: this should likely be the API
    der = cubic_spline_interpolation_first_derivatives(x, y)
    return cyclic_interp_curve(x, y, der)


def test_non_periodic():
    x = np.arange(8)
    y = np.arange(8)
    
    with pytest.raises(ValueError):
        make_spline(x, y)     # first and last points differ
    
    

def test_random(): 
    rndm = np.random.RandomState(1234)
    npts = 8
    x = np.sort(rndm.uniform(size=8))
    y = np.random.uniform(size=8)
    y[-1] = y[0]

    spl = make_spline(x, y)
    ynew = [spl(_) for _ in x]
    assert_allclose(ynew, y, atol=1e-12)
