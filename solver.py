import numpy as np
import scipy.sparse as ss
from kernels import Gaussian
from kernels import Derivative_Gaussian
import scipy.sparse.linalg as solv 
import time
import math

def MV_kronprod(krons,b):
    '''
    This function will perform a vector matrix product between a packed kronecker matrix
    and a column vector
    
    Inputs:
        krons --> list of tensor matrices
        b     --> Column vector
    
    Outputs:
        x  --> column vector resulting from product between the unpacked kronecker product and
               vector b 
    '''
    x = b
    N = len(b)
    D = len(krons)
    
    for d in reversed(xrange(D)):
        ld = len(krons[d])
        X = x.reshape((ld,N/ld),order='f') 
        Z = np.dot(krons[d],X).T
        x = Z.reshape((-1,1),order='f')
    return x
    

def kron_MVM(W,K,y,noise):
    ''' 
    This function performs a Matrix-Matrix-Matrix-Vector multiplication
    K_SKI*y = WKW'y
    efficiently in O(N+m^2) time and memory
    
    Inputs:
        W --> Interpolation weights matrix in compressed row format
        K --> mxm grid kernel matrix
        y --> Nx1 target value vector
    
    Outputs:
        WKW'y --> for use in Conjugate Gradient method
    '''
    s= math.exp(-noise)    
    return W.dot(MV_kronprod(K,W.transpose().dot(y))) + (s**2)*y   


def Linear_CG(W,K,y,noise,tolerance=1e-6,maxiter=50000):
    ''' Conjugate Gradient method to solve for alpha:
    alpha = inverse(WkW')*y
    
    Inputs:
        W --> Nxm weight matrix in sparse matrix format
        K --> Packed Kronecker set of dimensional gram matrices of the inducing points
        y --> Nx1 Target values
        noise --> scalar that represents the noise
        tolerance --> Stopping criteria of residual
        maxiter --> Maximum number of iterations
          
    Outputs:
        alpha --> Nx1 solution to linear equation
        ind  --> indicator of successful convergence. 0 if converged, else maxiter.
    '''    
   
    # initialize alpha
    alpha = np.zeros((len(y),1))
    # Perform first iteration 
    r_last = y
    p = r_last
    norm_r_last = np.linalg.norm(r_last)**2
    
    mvm = kron_MVM(W,K,p,noise)
    a = norm_r_last/np.dot(p.T,mvm)
    alpha = alpha + a*p
    r = r_last-a*mvm
    norm_r = np.linalg.norm(r)**2 
    iter = 1
    
    
    while (iter <maxiter and norm_r**0.5>tolerance):
        
        r_last=r
        B = norm_r/norm_r_last
        p = r_last+B*p
        norm_r_last = norm_r
        
        mvm = kron_MVM(W,K,p,noise)
        a = norm_r_last/np.dot(p.T,mvm)
        alpha = alpha + a*p
        r = r_last-a*mvm  
        norm_r = np.linalg.norm(r)**2 
        iter+=1
    
    #get indicator flag
    ind = 0 if iter != maxiter else iter     
    return (alpha,ind)

def NonLinear_CG(x,y,params,tolerance=1e-3,maxiter = 200):
    N = len(x)
    
    # Calculate the gradient wrt to paramters
    K_y,dsig,dl,ds   = Derivative_Gaussian(x,x,params[0],params[1],params[2])
    inv_Ky = np.linalg.inv(K_y)
    alpha = np.dot(inv_Ky,y)
    d_sigma = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_Ky),dsig))
    d_l = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_Ky),dl))
    d_s = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_Ky),ds))
    
    r_last = np.array([[d_sigma],[d_l],[d_s]])
    p = r_last
    norm_r_last = np.linalg.norm(r_last)**2
    
    # Make a line search algorithm
    params = Backtracking(x,y,0.25,params,p)
    
    
    
    K_y,dsig,dl,ds   = Derivative_Gaussian(x,x,params[0],params[1],params[2])    
    inv_Ky = np.linalg.inv(K_y)
    
    alpha = np.dot(inv_Ky,y)
    d_sigma = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_Ky),dsig))
    d_l = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_Ky),dl))
    d_s = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_Ky),ds))
    
    r = np.array([[d_sigma],[d_l],[d_s]])
    norm_r = np.linalg.norm(r)**2 
   
   
    iter = 1    
    
    while (iter <maxiter and norm_r**0.5>tolerance):
        
        r_last=r
        B = norm_r/norm_r_last
        p = r_last+B*p
        norm_r_last = norm_r
        
        params = Backtracking(x,y,0.25,params,p)
        
        K_y,dsig,dl,ds   = Derivative_Gaussian(x,x,params[0],params[1],params[2])
        inv_Ky = np.linalg.inv(K_y)

        alpha = np.dot(inv_Ky,y)
        d_sigma = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_Ky),dsig))
        d_l = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_Ky),dl))
        d_s = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_Ky),ds))
        
        r = np.array([[d_sigma],[d_l],[d_s]])
        norm_r = np.linalg.norm(r)**2 
        
        iter+=1
    #get indicator flag
    ML =  -0.5*np.dot(y.T,inv_Ky).dot(y)-0.5*np.log(np.linalg.det(K_y))-0.5*N*np.log(2*math.pi)
 
    ind = 0 if iter != maxiter else iter     
    return (params,ML,ind)
    

def Backtracking(x,y,rho,params,p):
   
    K_y = Gaussian(x,x,params[0],params[1],params[2])
    inv_Ky = np.linalg.inv(K_y)
    step = 1/np.linalg.norm(p)
    ML_initial =  -0.5*np.dot(y.T,inv_Ky).dot(y)-0.5*np.log(np.linalg.det(K_y))
        
 
    par = params + step*p
    K_y = Gaussian(x,x,par[0],par[1],par[2])
    inv_Ky = np.linalg.inv(K_y) 
    ML =  -0.5*np.dot(y.T,inv_Ky).dot(y)-0.5*np.log(np.linalg.det(K_y))
    
    iter = 0 
    
    
    while ML[0][0]<ML_initial[0][0] and iter<=15:
        step *= rho
        par = params + step*p
        K_y = Gaussian(x,x,par[0],par[1],par[2])
        inv_Ky = np.linalg.inv(K_y)
        ML =  -0.5*np.dot(y.T,inv_Ky).dot(y)-0.5*np.log(np.linalg.det(K_y))
        
        iter+=1      
    
    return par   
    
    
if __name__ == '__main__':       
    data = np.load('Regression_data.npz')
    x = data['x']
    y = data['y']

    params = NonLinear_CG(x,y,tolerance=1e-2,maxiter = 1000)
    
    
                             
    