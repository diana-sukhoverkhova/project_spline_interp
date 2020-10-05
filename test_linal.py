import numpy as np
from numpy.testing import assert_allclose
import pytest

from cyclic_interpolate import (sherman_morrison_algorithm, thomas_algorithm)


def matrix_splitter(a, b, c, alpha, beta):
    size = b.shape[0]
    matr = np.zeros([size, size])
    for i in range(size):
        matr[i][i] = b[i]
    for i in range(size - 1):
        matr[i][i + 1] = c[i]
        matr[i + 1][i] = a[i]
    matr[0][-1] = alpha
    matr[-1][0] = beta
    return matr

# thomas_algorithm
# sherman_morrison_algorithm


def test_different_shape():
    a = np.arange(4)
    b = np.arange(5)
    c = np.arange(5)
    d = np.arange(5)

    with pytest.raises(ValueError):
        thomas_algorithm(a, b, c, d)

    with pytest.raises(ValueError):
        sherman_morrison_algorithm(a, b, c, d, 1, 0)


def test_none():
    a = None
    b = np.arange(6)
    c = np.arange(5)
    d = np.arange(6)

    with pytest.raises(Exception):
        thomas_algorithm(a, b, c, d)

    with pytest.raises(ValueError):
        sherman_morrison_algorithm(a, b, c, d, 1, 0)

    a = np.arange(5)
    b = None
    c = np.arange(5)
    d = np.arange(6)

    with pytest.raises(Exception):
        thomas_algorithm(a, b, c, d)

    with pytest.raises(ValueError):
        sherman_morrison_algorithm(a, b, c, d, 1, 0)

    a = np.arange(5)
    b = np.arange(6)
    c = None
    d = np.arange(6)

    with pytest.raises(Exception):
        thomas_algorithm(a, b, c, d)

    with pytest.raises(ValueError):
        sherman_morrison_algorithm(a, b, c, d, 1, 0)

    a = np.arange(5)
    b = np.arange(6)
    c = np.arange(5)
    d = None

    with pytest.raises(Exception):
        thomas_algorithm(a, b, c, d)

    with pytest.raises(ValueError):
        sherman_morrison_algorithm(a, b, c, d, 1, 0)


def test_wrong_matrix_size():
    a = np.arange(1)
    b = np.arange(2)
    c = np.arange(1)
    d = np.arange(2)

    with pytest.raises(Exception):
        thomas_algorithm(a, b, c, d)

    with pytest.raises(ValueError):
        sherman_morrison_algorithm(a, b, c, d, 1, 0)


def test_random():
    rndm = np.random.RandomState(1234)
    a = rndm.uniform(size=7)
    b = rndm.uniform(size=8)
    c = rndm.uniform(size=7)
    d = rndm.uniform(size=8)
    matr = matrix_splitter(a, b, c, 0, 0)
    x1 = np.linalg.solve(matr, d)
    x2 = thomas_algorithm(a, b, c, d)
    assert_allclose(x1, x2, atol=1e-14)

    rndm = np.random.RandomState(1234)
    a = rndm.uniform(size=7)
    b = rndm.uniform(size=8)
    c = rndm.uniform(size=7)
    d = rndm.uniform(size=8)
    alpha = np.random.random_sample()
    beta = np.random.random_sample()
    matr = matrix_splitter(a, b, c, alpha, beta)
    x1 = np.linalg.solve(matr, d)
    x2 = sherman_morrison_algorithm(a, b, c, d, alpha, beta)
    assert_allclose(x1, x2, atol=1e-14)
