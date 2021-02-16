from numpy.testing import assert_allclose
import numpy as np

def woodbury_test():
    '''
    Random elements in diagonal matrix with blocks in the
    left lower and right upper corners checking the 
    implementation of Woodbury algorithm.
    '''
    def randomize(shape):
        return np.random.random_sample(shape) * 20 - 10

    for k in range(3,32,2):
        np.random.seed(1234)
        n = 201
        offset = int((k-1) / 2)
        a = np.diagflat(randomize((1,n)))
        for i in range(1,offset+1):
            a[:-i,i:] += np.diagflat(randomize((1,n-i)))
            a[i:,:-i] += np.diagflat(randomize((1,n-i)))
        ur = randomize((offset,offset))
        a[:offset,-offset:] = ur
        ll = randomize((offset,offset))
        a[-offset:,:offset] = ll
        d = np.zeros((k,n))
        for i,j in enumerate(range(offset,-offset-1,-1)):
            if j < 0:
                d[i,:j] = np.diagonal(a,offset=j)
            else:
                d[i,j:] = np.diagonal(a,offset=j)
        b = randomize((1,n))
        np.allclose(woodbury(d,ll,ur,b.T,k),np.linalg.solve(a,b.T))