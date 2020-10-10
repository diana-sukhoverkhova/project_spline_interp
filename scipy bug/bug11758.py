import scipy.interpolate as ipt
import numpy as np
import matplotlib.pyplot as plt
from cyclic_interpolate import get_first_derivatives, CyclicInterpCurve


def per_spl(x, y):
     assert len(x) == len(y) == 3
     y[-1] = y[0]  # shortcut

     x, y = map(np.asarray, (x, y))
     h = x[1:] - x[:-1]
     m = (y[1:] - y[:-1]) / h
     s = (m / h).sum() / (1. / h).sum()
     #s = ((m / h)/ (1. / h)/2).sum()
     print('NEW S : ', s)
     return ipt.CubicHermiteSpline(x, y, [s, s, s])

'''
def per_spl_old(x, y):
     assert len(x) == len(y) == 3
     y[-1] = y[0]  # shortcut

     x, y = map(np.asarray, (x, y))
     h = x[1:] - x[:-1]
     m = (y[1:] - y[:-1]) / h
     s = (m / h).sum() / (1. / h).sum()
     print('OLD S : ', s)
     return ipt.CubicHermiteSpline(x, y, [s, s, s])

'''
n = 3
x = np.sort(np.random.random_sample(n)*10)
y = np.random.random_sample(n)*30
y[-1] = y[0]
t = get_first_derivatives(x, y)
print("Our :", t[0])
try:
    spline = CyclicInterpCurve(x, y, t)
except ZeroDivisionError as e:
    print(e)
vhs = np.vectorize(spline)
vhs1 = np.vectorize(per_spl(x, y))
#vhs2 = np.vectorize(per_spl_old(x, y))
#plt.figure(figsize=(10,10))
plt.scatter(x, y, marker='.',c='red')
tmp = np.arange(x[0],x[n-1],0.01)
plt.plot(tmp,vhs(tmp), label='our')
plt.plot(tmp,vhs1(tmp), label='new')
#plt.plot(tmp,vhs2(tmp), label='old')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()
for i in range(n):
    if not np.allclose(y[i],spline(x[i]),atol=1e-15):
        print("Exception")
