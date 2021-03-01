import numpy as np
from scipy.interpolate import _bspl
import scipy.linalg as sl

def circle_vector(x,l=1,r=1):
    '''
    returns vector of nodes on circle
    '''

    if len(x) < max(l, r):
        raise ValueError("Too few points to make knot vector")

    dx = np.diff(x)
    t = np.zeros(len(x) + l + r)
    t[:l] = [x[0] - sum(dx[-i:]) for i in range(l,0,-1)]
    t[l:-r] = x
    t[-r:] = [x[-1] + sum(dx[:i]) for i in range(1,r+1)]
    return t

def B(x, k, i, t):
    if k == 0:
        if t[i] <= x < t[i+1]:
            return 1.0
        return 0.0
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        c2 = 0.0
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2

def bspline(x, t, c, k):
    n = len(t) - k - 1
    assert (n >= k+1) and (len(c) >= n)
    return sum(c[i] * B(x, k, i, t) for i in range(n))


def woodbury(A, ur, ll, b, k):
    '''
    Implementation of Woodbury algorithm applied to banded
    matrices with two blocks in upper right and lower left
    corners.

    Parameters
    ----------
    A : 2-D array, shape(k, n)
        Matrix of diagonals of original matrix(see
        'solve_banded' documentation).
    ll : 2-D array, shape(bs,bs)
        Lower left block matrix.
    ur : 2-D array, shape(bs,bs)
        Upper right block matrix.
    b : 1-D array, shape(1,n)
        Vector of constant terms of the SLE.

    Returns
    -------
    c : 1-D array, shape(1,n)
        Solution of the original SLE.

    Notes
    -----
    SLE - system of linear equations.

    'n' should be greater than 'k', otherwise corner block
    elements will intersect diagonals.
    '''
    k_odd = (k + 1) % 2
    bs = int((k - 1) / 2) + k_odd
    n = A.shape[1] + 1
    U = np.zeros((n - 1, k - k % 2))
    V = np.zeros((k - k % 2, n - 1))  # V transpose

    # upper right

    U[:bs, :bs] = ur
    for j in range(bs):
        V[j, -bs + j] = 1

    # lower left

    U[-bs:, -bs:] = ll
    for j in range(bs):
        V[-bs + j, j] = 1

    with np.printoptions(precision=2, suppress=True):
        print(U, '\n', V)

    Z = sl.solve_banded((bs - k_odd, bs), A, U[:, 0])  # z0
    Z = np.expand_dims(Z, axis=0)

    for i in range(1, k - k % 2):
        zi = sl.solve_banded((bs - k_odd, bs), A, U[:, i])
        zi = np.expand_dims(zi, axis=0)
        Z = np.concatenate((Z, zi), axis=0)

    Z = Z.transpose()
    H = sl.inv(np.identity(k - k % 2) + V @ Z)

    y = sl.solve_banded((bs - k_odd, bs), A, b)
    c = y - Z @ (H @ (V @ y))

    return np.concatenate((c[-bs + k_odd:], c, c[: bs + 1]))

def make_matrix(x,y,t,k):
    '''
    Returns diagonals and block elements of the matrix
    formed by points (x, y) for coefficients for B-Spline
    curve.
    
    Parameters
    ----------
    x : 1-D array, shape(n,)
        x coordinate of points.
    y : 1-D array, shape(n,).
        y coordiante of points.
    t : 1-D array, shape(n+2*k,)
        Node vector.
    k : int
        Degree of B-splines.

    Returns
    -------
    A : 2-D array, shape(k,n-1)
        Diagonals of the original matrix of SLE.
    ur : 2-D array, shape(offset,offset)
        Upper right block of the original matrix of SLE.
    ll : 2-D array, shape(offset,offset)
        Lower left block of the original matrix of SLE.

    Notes
    -----
    SLE - system of linear equations.
    
    'n' should be greater than 'k', otherwise corner block
    elements will intersect diagonals.

    offset = (k - 1) / 2
    '''
    if k % 2 == 0:
        raise ValueError("Degree of B-spline should be odd")
    n = x.size
    yc = np.copy(y)
    yc = yc[:-1]
    A = np.zeros((k, n - 1))
    for i in range(n-1):
        A[:,i] = _bspl.evaluate_all_bspl(t, k, x[i], i + k)[:-1][::-1]
    offset = int((k-1)/2) + (k + 1) % 2
    ur = np.zeros((offset,offset))
    ll = np.zeros((offset,offset))
    for i in range(1,offset + 1):
        A[offset - i] = np.roll(A[offset - i],i)
        if k % 2 == 1 or i < offset:
            A[offset + i] = np.roll(A[offset + i],-i)
            ur[-i:,i-1] = np.copy(A[offset + i,-i:])
        ll[-i,:i] = np.copy(A[offset - i,:i])
    ur = ur.T
    for i in range(1,offset):
        ll[:,i] = np.roll(ll[:,i],i)
        ur[:,-i-1] = np.roll(ur[:,-i-1],-i)
    return A, ur, ll
