import GP_Regression
import numpy as np
import kernels 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import solver

if __name__ == '__main__':
        
    # sample training points from a normal distribution with covariance 4I
    
    #TO DO : set as variables (scale, hyperparameters
    
    N = 10000
    
    sigma = 3
    l     = -1
    s     = 8
    
    
    x1 = np.sort(np.random.normal(scale=4,size=(1,N))).reshape(N,1)
    x2 = np.sort(np.random.normal(scale=4,size=(1,N))).reshape(N,1)
    x = np.hstack((x1,x2))
    
    Ky = kernels.Gaussian(x,x,sigma,l,s)
    L = np.linalg.cholesky(Ky)
    v = np.random.normal(scale=1,size=(N,1))
  
    y = np.dot(L,v)             
    
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x[:,0],x[:,1],y,c='b')
    
    plt.figure(2)
    plt.clf()
    plt.plot(x[:,0],y)
    
    plt.figure(3)
    plt.clf()
    plt.plot(x[:,1],y)
    
    plt.figure(4)
    plt.clf()
    plt.scatter(x[:,0],x[:,1])
    
    
    parameters={ 'dataset':'Modified',
                 'n':10000,
                 's':0,
                 'sigma': 0,
                 'l': 1 ,
                 'kernel': 'SE',
                 'x' : x,
                 'y' : y,
                 'X' : x,
                 }
    
    sample = GP_Regression.GPRegression(parameters)
    sample.Inducing_Points()
    sample.optimizeHyperparameters()
    
    sample.KISS_GP() 
    
    noise = math.exp(-sample.s)
    cholk = [np.linalg.cholesky(sample.grid.Kd[0]+noise*np.eye(100)), np.linalg.cholesky(sample.grid.Kd[1]+noise*np.eye(100))]
    
    
   
    
    ys = sample.grid.W.dot(solver.MV_kronprod(cholk,v))
    