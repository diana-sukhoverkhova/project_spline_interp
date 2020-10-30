import numpy as np
import matplotlib.pyplot as plt

accuracy = 1e-14


def make_spline(x, y):
    # A convenience wrapper for the ctor.
    der = get_first_derivatives(x, y)
    return CyclicInterpCurve(x, y, der)


class CyclicInterpCurve:
    """
    Cubic interpolating function in Hermite form with periodic
    boundary conditions.

    Parameters
    ----------
    x : 1-D array, shape (n,)
        Values of x - coordinate of a given set of points.
    y : 1-D array, shape (n,)
        Values of y - coordinate of a given set of points.
    der : 1-D array, shape (n,)
        First derivatives, found from condition of equality
        neighboring derivatives and from a given set of points (x, y)
        for building up cubic polynomial in Hermite form.

    Attributes
    ----------
    n : int
        Number of given points.
    c : 2-D array, shape (4, n)
        Array of coefficients of polynomials
        ''sum(csp.c[m,i]*(xp-x[i])**(3-m) for m in range(4))''

    Methods
    -------
    __call__
    hermite_cubic_spline
    second_derivatives
    """

    def __init__(self, x, y, der):
        if x is None or y is None or der is None:
            raise ValueError("Cannot initialize an instance because some parameters is None")
        if not all(u < v for u, v in zip(x, x[1:])):
            raise ValueError("x should strictly increase")
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
        self.der = np.copy(der)  # vector of derivatives
        self.c = np.zeros([4, self.n-1])
        for i in range(self.n-1):
            h = x[i+1]-x[i]
            m = (y[i+1]-y[i])/h
            self.c[3, i] = y[i]
            self.c[2, i] = der[i]
            self.c[1, i] = (3*m-2*der[i]-der[i+1])/h
            self.c[0, i] = -(2*m-der[i]-der[i+1])/(h*h)

    def __call__(self, xnew):
        """
        returns the value of the spline at 'xnew'
        """
        period = self.x[-1] - self.x[0]
        if xnew < self.x[0]:
            n = int((self.x[0] - xnew) / period) + 1
            xnew += n * period
        elif xnew > self.x[-1]:
            n = int((xnew - self.x[-1]) / period) + 1
            xnew -= n * period
        return self.hermite_cubic_spline(xnew)

    def hermite_cubic_spline(self, t):
        """
        Build up spline function - cubic polynomial in Hermite form
        and returns value of this spline in t.
        """
        #i = np.argmax(self.x > t) - 1
        i = max(np.argmax(self.x >= t) - 1, 0)
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

    def second_derivatives(self):
        s = self.der
        n = self.n
        h = np.zeros(n - 1)
        m = np.zeros(n - 1)
        p = np.zeros(n)

        for i in range(self.n - 1):
            h[i] = 1 / (self.x[i + 1] - self.x[i])
            m[i] = (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i])
        for i in range(self.n - 1):
            p[i] = -2 * h[i] * (s[i + 1] - m[i]) - 4 * h[i] * (s[i] - m[i])
        p[-1] = p[0]
        return p


def get_first_derivatives(x, y):
    """
    Return 1-D array of first derivatives.

    Given two 1-D arrays ''x'' and ''y'', returns first derivatives in
    points '(x, y)'.

    Parameters
    ----------
    x : numpy.array
        Represents the x-coordinates of a set of datapoints.
    y : numpy.array
        Represents the y-coordinates of a set of datapoints, i.e., f(`x`).

    Returns
    -------
    numpy.array
        First derivatives in points '(x, y)'

    Notes
    -----
    Array ''x'' should be sorted.

    First and last y-coordinates should be equal to reach the periodic
    condition.

    Number of point should be larger than 3 because minimally 4 points
    can form the interpolate function.

    The case of 3 points is performed with the same algorithm
    made up manually
    """
    x = np.array(x)
    y = np.array(y)
    if x is None or y is None:
        raise ValueError("Some of arguments are None")
    if not all(u < v for u, v in zip(x, x[1:])):
        raise ValueError("x should strictly increase")
    if x.shape[0] <= 2:
        raise ValueError(f"{x.shape[0]} points are not enough to interpolate", x.shape[0])
    if not np.allclose(y[0], y[-1], atol=accuracy):
        raise ValueError("Function does not match at first and last points (periodic condition)", y[0], y[-1])
    if x.shape != y.shape:
        raise ValueError(f"x and y have different sizes ({x.shape[0]} != {y.shape[0]})", x.shape[0], y.shape[0])
    if x.shape[0] == 3:
        h = x[1:] - x[:-1]
        m = (y[1:] - y[:-1]) / h
        s = (m / h).sum() / (1. / h).sum()
        return [s, s, s]
    n = x.shape[0] - 2  # n + 2 dots given, splits into n + 1 intervals
    s = np.zeros(n + 1, dtype=float)
    r = np.zeros(n + 1, dtype=float)
    # a, b, c - 3 arrays that form a tridiagonal matrix
    a = np.zeros(n, dtype=float)
    b = np.zeros(n + 1, dtype=float)
    c = np.zeros(n, dtype=float)
    d = np.zeros(n + 1, dtype=float)  # vector of constant terms
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
    d[0] = 3*(s[n] * r[n] + s[0] * r[0])
    d[n] = 3*(s[n - 1] * r[n - 1] + s[n] * r[n])
    b[0] = 2 * (s[n] + s[0])
    c[0] = s[0]
    a[n - 1] = s[n - 1]
    b[n] = 2 * (s[n - 1] + s[n])

    # as well as first derivatives are equal we add the first element of solution to the end
    result = sherman_morrison_algorithm(a, b, c, d, s[n], s[n])
    return np.append(result, result[0])


def sherman_morrison_algorithm(a, b, c, r, alpha, beta):
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
    if a.shape != c.shape or a.shape[0] + 1 != b.shape[0] or b.shape != r.shape:
        raise ValueError(f"Vectors a({a.shape[0]}), b({b.shape[0]}), c({c.shape[0]}),"
                         f"d({r.shape[0]}) have incompatible sizes",
                         a.shape, b.shape, c.shape, r.shape)
    ac, bc, cc = np.copy(a), np.copy(b), np.copy(c)
    n = b.shape[0]  # size
    u = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)
    u[0] = alpha
    u[-1] = beta
    v[0] = 1
    v[-1] = 1
    bc[0] -= alpha
    bc[-1] -= beta

    w = thomas_algorithm(ac, bc, cc, r)
    z = thomas_algorithm(ac, bc, cc, u)
    x = np.zeros(n, dtype=float)
    if w.shape != v.shape or z.shape != v.shape:
        raise ValueError("Wrong output from Thomas algorithm")
    vw = v.dot(w)
    vz = v.dot(z)
    if np.allclose(vz, -1, atol=accuracy):
        raise ZeroDivisionError("Division by zero")
    for i in range(n):
        x[i] = w[i] - z[i] * vw / (1 + vz)
    return x


def thomas_algorithm(a, b, c, d):
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
    if a.shape != c.shape or a.shape[0] + 1 != b.shape[0] or b.shape != d.shape:
        raise ValueError(f"Vectors a({a.shape[0]}), b({b.shape[0]}), c({c.shape[0]}),"
                         f"d({d.shape[0]}) have incompatible sizes",
                         a.shape, b.shape, c.shape, d.shape)
    n = b.shape[0]
    if n < 2:
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

    x = np.zeros([n, ], dtype=float)
    x[n - 1] += dc[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] += dc[i] - cc[i] * x[i + 1]
    return x
