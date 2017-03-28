import numpy as np
import Kernels
import math

N=20

x1 = np.sort(25*np.random.rand(1,N)-25*np.random.rand(1,N)).reshape(N,1)
x2 =  np.sort(25*np.random.rand(1,N)-25*np.random.rand(1,N)).reshape(N,1)
x1s = np.linspace(-28,28,num=300).reshape(300,1)
x2s = np.linspace(-28,28,num=300).reshape(300,1)
x = np.hstack((x1,x2))
xs=np.hstack((x1s,x2s))
y= x1**2 - 10*x1*(np.sin(x2))**3 + np.random.normal(scale=10,size=(N,1))


hyp = np.array([[-3.5],[-1.2],[5.0]])

K = Kernels.Gaussian_Kernel(x1,x2,hyp,n=True)

alpha = np.dot(np.linalg.inv(K),y)

alpha1 = np.dot(np.linalg.inv(K+0.1*np.eye(N)),y)

f1 = np.dot(y.T,alpha)
f2 = np.dot(y.T,alpha1)

sigma = math.exp(-hyp[0])
rank_fix = (sigma**2)/1e6

diff = f1-f2
print(diff)

a = Kernels.D_Gaussian(x,y,hyp,0,n=True)
b = Kernels.D_Gaussian(x,y,hyp,rank_fix,n=True)

g1 = a[0]
f1 = a[1]
g2 = b[0]
f2 = b[1]