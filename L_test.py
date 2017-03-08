import numpy as np
import GP_Regression as GPR
import kernels 
import Kronecker
import time
import math
import matplotlib.pyplot as plt

def kron(a,b):
    I,J = a.shape
    Q,W = b.shape
    
    k = np.zeros((I*Q,J*W))
    
    for i in xrange(I):
        for j in xrange(J):
            k[(i*Q):((i+1)*Q),(j*W):((j+1)*W)] = a[i,j]*b
    return k


def test_parameters(N,dim,samples=20,var=25, inducing_points = [10,15]):
    x=np.array([])
    for n in xrange(dim):
        x1 = (np.random.normal(scale=var,size=(1,N))).reshape(N,1)
        x = (np.hstack((x,x1)) if x.size else x1)

    xs=0
    y= 0
    
    parameters={ 'dataset':'Modified',
                 'n':N,
                 's': 7.0,
                 'sigma': 0.0,
                 'l': -4 ,
                 'kernel': 'SE',
                 'x' : x,
                 'y' : y,
                 'X' : xs,
                 }
    
    
    l = np.linspace(3,-4,num=samples)
    error_l = []
    sigma = np.linspace(3,-3,num=samples)
    error_sigma = []
    
    plt.figure(1)
    plt.clf()
    for j in inducing_points:
        for i in sigma:
            parameters['sigma'] = i 
            grid = Kronecker.tensor_grid(x,[j]*dim)   
            grid.generate(parameters)
            grid.SKI(interpolation='cubic')
            noise = math.exp(-parameters['s'])
            grid.K = grid.Kd[dim-1]
            for d in reversed(xrange(dim-1)):
                grid.K = kron(grid.Kd[d],grid.K)
            K_SKI = grid.W.dot((grid.W.dot(grid.K)).T).T + (noise**2)*np.eye(N)   
            K = kernels.Gaussian(x,x,parameters['sigma'],parameters['l'],parameters['s'])
            error_sigma.append(np.linalg.norm(K-K_SKI))
        error_sigma = (1.0/np.linalg.norm(K))*np.array(error_sigma)
        plt.plot(np.exp(-sigma),error_sigma,label='m=%i'%(j))
        error_sigma = []
    plt.title('Reconstruction Error in %iD' %(dim))
    plt.xlabel('sigma')
    plt.ylabel('Reconstruction Error')
    plt.legend(loc=2)
    plt.savefig('%iD_sigma.png' %(dim)) 
    
    
    parameters['sigma'] = 0.0
    plt.figure(2) 
    plt.clf()
    for j in inducing_points:
        for i in l:
            parameters['l'] = i 
            grid = Kronecker.tensor_grid(x,[j]*dim)   
            grid.generate(parameters)
            grid.SKI(interpolation='cubic')
            noise = math.exp(-parameters['s'])
            grid.K = grid.Kd[dim-1]
            for d in reversed(xrange(dim-1)):
                grid.K = kron(grid.Kd[d],grid.K)
            K_SKI = grid.W.dot((grid.W.dot(grid.K)).T).T + (noise**2)*np.eye(N)   
            K = kernels.Gaussian(x,x,parameters['sigma'],parameters['l'],parameters['s'])
            error_l.append(np.linalg.norm(K-K_SKI))
        error_l = (1.0/np.linalg.norm(K))*np.array(error_l)
        plt.plot(np.exp(-l),error_l,label='m=%i'%(j))
        error_l= []   
    plt.title('Reconstruction Error in %iD' %(dim))
    plt.xlabel('Characteristic Length')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.savefig('%iD_Length.png' %(dim))    


def test_dims(N,inducing_points=10,samples=15,var=25, dims = [1,2,3,4]):
    
    L = np.linspace(3,-4,num=samples)
    plt.figure(1) 
    plt.clf()
    for dim in dims:  
        error_l = []
        x=np.array([])
        for n in xrange(dim):
            x1 = (np.random.normal(scale=var,size=(1,N))).reshape(N,1)
            x = (np.hstack((x,x1)) if x.size else x1)
    
        xs=0
        y= 0
        
        parameters={ 'dataset':'Modified',
                     'n':N,
                     's': 7.0,
                     'sigma': 0.0,
                     'l': -4 ,
                     'kernel': 'SE',
                     'x' : x,
                     'y' : y,
                     'X' : xs,
                     } 
        grid = Kronecker.tensor_grid(x,[inducing_points]*dim)
        grid.generate(parameters)  
        grid.SKI(interpolation='ModifiedSheppard') 
        for l in L:
            parameters['l'] = l   
            grid.generate(parameters)
            noise = math.exp(-parameters['s'])
            grid.K = grid.Kd[dim-1]
            for d in reversed(xrange(dim-1)):
                grid.K = kron(grid.Kd[d],grid.K)
            K_SKI = grid.W.dot((grid.W.dot(grid.K)).T).T + (noise**2)*np.eye(N)   
            K = kernels.Gaussian(x,x,parameters['sigma'],parameters['l'],parameters['s'])
            error_l.append(np.linalg.norm(K-K_SKI))
        error_l = (1.0/np.linalg.norm(K))*np.array(error_l)
        plt.plot(np.exp(-L),error_l,label='%iD'%(dim))
        error_l= []   
    
    plt.title('Dimensional Analysis of Reconstruction Error')
    plt.xlabel('Characteristic Length')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.savefig('Dimensional_ReconstructionError_L.png')   

if __name__ == '__main__':
    test_dims(1000)
