import numpy as np
import time

def kron(a,b):
    I,J = a.shape
    Q,W = b.shape
    
    k = np.zeros((I*Q,J*W))
    
    for i in xrange(I):
        for j in xrange(J):
            k[(i*Q):((i+1)*Q),(j*W):((j+1)*W)] = a[i,j]*b
    return k


a = np.random.normal(scale=5,size=(100,100))

b = np.random.normal(scale=5,size=(100,100))

start = time.time()
c = np.kron(a,b)
end = time.time()
print('done in %.8f seconds' %(end-start))

start = time.time()
d = kron(a,b)
end = time.time()
print('done in %.8f seconds' %(end-start))
