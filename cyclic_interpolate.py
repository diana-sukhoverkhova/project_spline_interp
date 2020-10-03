import numpy as np

accuracy = 1e-14


class CyclicInterpCurve:
    """
    Cubic interpolating function in Hermite form with periodic
    boundary conditions.

    Parameters
    ----------
        x : 1-D array, shape (k,)
            Values of x - coordinate of a given set of points.
        y : 1-D array, shape (k,)
            Values of y - coordinate of a given set of points.
        der : 1-D array, shape (k,)
            First derivatives, found from condition of equality
            neighboring derivatives and from a given set of points (x, y)
            for building up cubic polynomial in Hermite form.

    Attributes
    ----------
    n : int
        Number of given points.

    Methods
    -------
    __call__
    hermit_cubic_spline

    """
    def __init__(self, x, y, der):
        self.x = x
        self.y = y
        self.n = len(x)
        self.der = np.copy(der)  # vector of derivatives

    def __call__(self, xnew):
        """
        returns the value of the spline at 'xnew'
        """
        if self.x[0] > xnew or self.x[-1] < xnew:
            raise ValueError(f'xnew not in ({x[0]},{x[-1]})', xnew)
        return self.hermit_cubic_spline(xnew)

    def hermit_cubic_spline(self, t):
        """
        Build up spline function - cubic polynomial in Hermite form
        and returns value of this spline in t.
        """
        i = np.argmax(self.x > t) - 1

        xa = self.x[i]
        xb = self.x[i + 1]
        ya = self.y[i]
        yb = self.y[i + 1]
        sa = self.der[i]
        sb = self.der[i + 1]
        if np.allclose(xb, xa, atol=accuracy):
            raise ZeroDivisionError("Division by zero", xa, xb)
        h = xb - xa
        m = (yb - ya) / h
        p1 = yb * (t - xa) / h + ya * (xb - t) / h
        p2 = (t - xa) / h * (xb - t) / h
        p3 = (m - sb) * (t - xa) + (sa - m) * (xb - t)
        return p1 + p2 * p3


def cubic_spline_interpolation_first_derivatives(x, y):
    """
    Return 1-D array of first derivatives.

    Given two 1-D arrays 'x' and 'y', returns first derivatives in
    points ''(x, y)''.

    Parameters
    ----------
    x : numpy.array
        Represents the x-coordinates of a set of datapoints.
    y : numpy.array
        Represents the y-coordinates of a set of datapoints, i.e., f(`x`).

    Returns
    -------
    numpy.array
        First derivatives in points ''(x, y)''

    Notes
    -----
    Array 'x' should be sorted.

    First and last y-coordinates should be equal to reach the periodic
    condition.

    Number of point should be greater than 2 because minimally 3 points
    can form the interpolate function.
    """
    if x is None or y is None:
        raise Exception("Some of arguments are None")
    if not np.allclose(x, np.sort(x), atol=accuracy):
        raise ValueError("x should be a sorted array", x)
    if x.shape[0] < 2:
        raise ValueError(f"{x.shape[0]} points are not enough to interpolate", x.shape[0])
    if not np.allclose(y[0], y[-1], atol=accuracy):
        raise ValueError("Function does not match at first and last points (periodic condition)", y[0], y[-1])
    if x.shape != y.shape:
        raise ValueError(f"x and y have different sizes ({x.shape[0]} != {y.shape[0]})", x.shape[0], y.shape[0])
    n = x.shape[0] - 2  # n + 2 dots given, splits into n + 1 intervals
    s = np.zeros(n + 1)
    r = np.zeros(n + 1)
    # a, b, c - 3 arrays that form a tridiagonal matrix
    a = np.zeros(n)
    b = np.zeros(n + 1)
    c = np.zeros(n)
    d = np.zeros(n + 1)  # vector of constant terms
    for i in range(n + 1):
        if np.allclose(x[i + 1], x[i], atol=accuracy):
            raise ZeroDivisionError("Division by zero", x[i], x[i + 1])
        s[i] = 1 / (x[i + 1] - x[i])
        r[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    for i in range(1, n):
        a[i - 1] = s[i - 1]
        b[i] = 2 * (s[i - 1] + s[i])
        c[i] = s[i]
        d[i] = 3 * (s[i - 1] * r[i - 1] + s[i] * r[i])
    d[0] = s[n] * r[n] + s[0] * r[0]
    d[n] = s[n - 1] * r[n - 1] + s[n] * r[n]
    b[0] = 2 * (s[n] + s[0])
    c[0] = s[0]
    a[n - 1] = s[n - 1]
    b[n] = 2 * (s[n - 1] + s[n])
    # as well as first derivatives are equal we add the first element of solution to the end
    result = Sherman_Morrison_algorithm(a, b, c, d, s[n], s[n])
    return np.append(result, result[0])


def Sherman_Morrison_algorithm(a, b, c, r, alpha, beta):
    """
    Implementation of Sherman-Morrison's algorithm.

    Returns solution to the system of linear equations with almost
    tridiagonal matrix.

    Parameters
    ----------
    a : np.array
        lower diagonal elements of tridiagonal matrix of the SLE.
    b : np.array
        diagonal elements of tridiagonal matrix of the SLE.
    c : np.array
        upper diagonal elements of tridiagonal matrix of the SLE.
    r : np.array
        array of constant terms of the SLE.
    alpha : float
        The right corner element of matrix of the SLE.
    beta : float
        The left corner element of matrix of the SLE.

    Returns
    -------
    x : np.array
        Solution of the SLE.

    Notes
    -----
    SLE - system of linear equations.

    The minimum size of matrix of SLE is 3 because
    tridiagonal matrix can not be defined otherwise.
    """
    if a is None or b is None or c is None or r is None:
        raise Exception("Some of arguments are None")
    if b.shape[0] <= 2:
        raise ValueError("Matrix size is not enough to interpolate")
    if a.shape != c.shape or a.shape[0] + 1 != b.shape[0]:
        raise ValueError(f"Vectors a({a.shape[0]}), b({b.shape[0]}), c({c.shape[0]}) have incompatible sizes",
                         a.shape, b.shape, c.shape)
    ac, bc, cc = np.copy(a), np.copy(b), np.copy(c)
    n = b.shape[0]  # size
    u = np.zeros(n)
    v = np.zeros(n)
    u[0] = alpha
    u[-1] = beta
    v[0] = 1
    v[-1] = 1
    bc[0] -= alpha
    bc[-1] -= beta

    w = Thomas_algorithm(ac, bc, cc, r)
    z = Thomas_algorithm(ac, bc, cc, u)
    x = np.zeros(n)
    if w.shape != v.shape or z.shape != v.shape:
        raise ValueError("Wrong output from Thomas algorithm")
    vw = v.dot(w)
    vz = v.dot(z)
    if np.allclose(vz, -1, atol=accuracy):
        raise ZeroDivisionError("Division by zero")
    for i in range(n):
        x[i] = w[i] - z[i] * vw / (1 + vz)
    return x


def Thomas_algorithm(a, b, c, d):
    """
    Implementation of Thomas algorithm.

    Parameters
    ----------
    a : 1-D array, shape (m-1,)
        Vector of elements x_{i,i-1} of a tridiagonal matrix.
    b : 1-D array, shape (m,)
        Main diagonal of a tridiagonal matrix.
    c : 1-D array, shape (m-1,)
        Vector of elements x_{i-1,i} of a tridiagonal matrix.
    d : 1-D array, shape (n,)
        Vector of constants terms of a system of linear equations.

    Returns
    -------
    х : 1-D array
        Solution to a matrix equation.

    Notes
    -----
    If original tridiagonal matrix has shape n x m, then
    length of vector b is m as a main diagonal,
    lengths of vectors a and c is m - 1,
    and length of vector d is n.

    """
    ac, bc, cc, dc = np.copy(a), np.copy(b), np.copy(c), np.copy(d)
    if ac is None or bc is None or cc is None or dc is None:
        raise Exception("Error of copying")
    if a.shape != c.shape or a.shape[0] + 1 != b.shape[0]:
        raise ValueError(f"Vectors a({a.shape[0]}), b({b.shape[0]}), c({c.shape[0]}) have incompatible sizes",
                         a.shape, b.shape, c.shape)
    n = b.shape[0]
    if n <= 2:
        raise ValueError("Matrix size is not enough to solve linear system")

    # stage 1

    if np.allclose(bc[0], 0, atol=accuracy):
        raise ZeroDivisionError("Division by zero", bc[0])
    beta = bc[0]
    bc[0] = 1
    cc[0] = cc[0] / beta
    dc[0] = dc[0] / beta

    for i in range(1, n - 1):
        if np.allclose(bc[i], cc[i - 1] * ac[i - 1], atol=accuracy):
            raise ZeroDivisionError("Division by zero", bc[i], cc[i - 1] * ac[i - 1])
        alpha = (bc[i] - cc[i - 1] * ac[i - 1])
        beta = ac[i - 1]
        ac[i - 1] -= ac[i - 1] * bc[i - 1]
        bc[i] = (bc[i] - cc[i - 1] * beta) / alpha
        cc[i] = cc[i] / alpha
        dc[i] = (dc[i] - dc[i - 1] * beta) / alpha

    i = n - 1
    if np.allclose(bc[i], cc[i - 1] * ac[i - 1], atol=accuracy):
        raise ZeroDivisionError("Division by zero", bc[i], cc[i - 1] * ac[i - 1])
    alpha = (bc[i] - cc[i - 1] * ac[i - 1])
    beta = ac[i - 1]
    ac[i - 1] -= ac[i - 1] * bc[i - 1]
    bc[i] = (bc[i] - cc[i - 1] * beta) / alpha
    dc[i] = (dc[i] - dc[i - 1] * beta) / alpha

    # stage 2

    x = np.zeros([n, ])
    x[n - 1] += dc[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] += dc[i] - cc[i] * x[i + 1]
    return x
