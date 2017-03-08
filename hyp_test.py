import GP_Regression
import numpy as np
import kernels 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import solver
import math
import gc
import time


if __name__ == '__main__':
    gc.collect()    
    # sample training points from a normal distribution with covariance 4I
    
    #TO DO : set as variables (scale, hyperparameters
    
    N = 5000
    M = 200
    
    sigma = 2.0
    l     = -3.0
    s     = 7.0
    
    x1 = np.sort(np.random.normal(scale=2,size=(1,N))).reshape(N,1)
    x2 = np.sort(np.random.normal(scale=2,size=(1,N))).reshape(N,1)
    x = np.hstack((x1,x2))
    
    Ky = kernels.Gaussian(x,x,sigma,l,s)
    L = np.linalg.cholesky(Ky)
    v = np.random.normal(scale=1,size=(N,1))
  
    y = np.dot(L,v)             
    
    plt.figure(1)
    plt.clf()
    plt.plot(x[:,0],y)
    
    plt.figure(2)
    plt.clf()
    plt.plot(x[:,1],y)
     
    parameters1={ 'dataset':'Modified',
                 'n': N,
                 's':10,
                 'sigma': -2.0,
                 'l': -2,
                 'kernel': 'SE',
                 'x' : x,
                 'y' : y,
                 'X' : x,
                 }
    
    sample1 = GP_Regression.GPRegression(parameters1)
    start = time.time()
    #sample1.optimizeHyperparameters(kron=False)  
    end = time.time()
    print('GP kernel hyperparameter optimization done in %.8f seconds' %(end-start))
    K_y = kernels.Gaussian(sample1.x,sample1.x,sample1.sigma,sample1.l,sample1.s)
    LGP = np.linalg.cholesky(K_y)
    ygp = np.dot(LGP,v)     

    plt.figure(3)
    plt.clf()
    plt.plot(sample1.x[:,0],ygp)

    plt.figure(4)
    plt.clf()
    plt.plot(sample1.x[:,1],ygp) 


    parameters2={ 'dataset':'Modified',
                 'n': N,
                 's':5.0,
                 'sigma': 4.0,
                 'l': -5.0,
                 'kernel': 'SE',
                 'x' : x,
                 'y' : y,
                 'X' : x,
                 }
    
    sample = GP_Regression.GPRegression(parameters2)
    sample.Inducing_Points()
    start = time.time()
    sample.optimizeHyperparameters()
    end = time.time()
    print('KISS-GP kernel hyperparameter optimization done in %.8f seconds' %(end-start))
    sample.KISS_GP()
    
    noise = math.exp(-sample.s)
    
    K = np.kron(sample.grid.Kd[0],sample.grid.Kd[1])
    K_SKI = sample.grid.W.dot((sample.grid.W.dot(K)).T).T + (noise**2)*np.eye(N) 
    Lski = np.linalg.cholesky(K_SKI)
    yt = np.dot(Lski,v) 

    plt.figure(5)
    plt.clf()
    plt.plot(sample.x[:,0],yt)

    plt.figure(6)
    plt.clf()
    plt.plot(sample.x[:,1],yt)  
    

    