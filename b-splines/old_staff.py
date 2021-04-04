import numpy as np
from scipy.interpolate import _bspl

def find_left(ar,val):
    assert min(ar) <= val
    return len(ar[ar <= val]) - 1

def make_full_matr(x, y, t, k):
    n = len(x)
    matr = np.zeros((n+k,n+k))
    for i in range(n):
        matr[i + k - 1,i:i+k+1] = _bspl.evaluate_all_bspl(t, k, x[i], find_left(t,x[i]))
    for i in range(k-1):
        matr[i,:k + 1] = _bspl.evaluate_all_bspl(t, k, x[0], find_left(t,x[0]), nu=i+1)
        matr[i, -k-1:] = -_bspl.evaluate_all_bspl(t, k, x[-1], find_left(t,x[-1]), nu=i+1)
    matr = matr[:-1,:-1]
    b = np.zeros_like(matr[:,0])
    b[k-1:] = y
    return matr,b