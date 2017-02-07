import numpy as np
import scipy.sparse as ss
from kernels import Gaussian
import scipy.sparse.linalg as solv 
import time

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
    

def kron_MVM(W,K,y,sigma):
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
        
    return W.dot(MV_kronprod(K,W.transpose().dot(y))) + sigma*y   


def CG(W,K,y,sigma=1e-5,tolerance=1e-6,maxiter=50000):
    ''' Conjugate Gradient method to solve for alpha:
    alpha = inverse(WkW')*y
    
    Inputs:
        W --> Nxm weight matrix in sparse matrix format
        K --> Packed Kronecker set of dimensional gram matrices of the inducing points
        y --> Nx1 Target values
        sigma --> scalar that represents the noise
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
    
    mvm = kron_MVM(W,K,p,sigma)
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
        
        mvm = kron_MVM(W,K,p,sigma)
        a = norm_r_last/np.dot(p.T,mvm)
        alpha = alpha + a*p
        r = r_last-a*mvm  
        norm_r = np.linalg.norm(r)**2 
        iter+=1
    
    #get indicator flag
    ind = 0 if iter != maxiter else iter     
    return (alpha,ind)

if __name__ == '__main__':       
    d = np.random.normal(loc=0.5,scale=0.15,size=(1000,))
    ind1 = np.array(xrange(1000)).reshape(1000,)   
    ind2 = np.random.randint(0,high=1000,size=(1000,))
    
    x1= np.array([1,2,3,4,5,6,7,8,9,10]).reshape(10,1)
    x2 = np.array([0.2,1,1.8,2.6,3.4,5.2,6,6.8,7.6,8.4]).reshape(10,1)
    x3 = np.array([2.3,2.6,2.9,3.2,3.5,3.8,4.1,4.4,4.7,5]).reshape(10,1)
    
    W  = ss.csr_matrix((d,(ind1,ind2)),shape=(1000,1000))
    K1 = Gaussian(x1,x1,1,1)
    K2 = Gaussian(x2,x2,1,1)
    K3 = Gaussian(x3,x3,1,1)
       
    Kd = [K1,K2,K3]
    K = np.kron(K1,np.kron(K2,K3))
    
    K_SKI = (W.dot(K.T)).T
    K_SKI = W.dot(K_SKI)
    K_SKI = K_SKI + 0.0001*np.eye(1000)
    
    y = np.random.normal(loc=5,scale=1,size=(1000,1))
    
    start = time.time()
    alpha_test = CG(W,Kd,y,maxiter=2000)
    end = time.time()
    print('Done in %.9f seconds' %(end-start))
     
    
    start = time.time()
    alpha = solv.cg(K_SKI,y,tol=1e-6,maxiter=2000)
    end = time.time()
    print('Done in %.9f seconds' %(end-start))

                             
    