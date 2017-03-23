import numpy as np

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
    K_SKI*y = (WKW' + spherical_noise)*y
    efficiently in O(N+m^2) time and memory
    
    Inputs:
        W     --> Interpolation weights matrix in compressed row format
        K     --> mxm grid kernel matrix
        y     --> Nx1 target value vector
        noise --> scalar value of the noise
    
    Outputs:
        (WKW' + spherical_noise)*y --> for use in Conjugate Gradient method
    '''
        
    return W.dot(MV_kronprod(K,W.transpose().dot(y))) + (noise**2)*y  


def KSKI_Unpack(W,K,noise):
    '''
     Use this function to unpack KSKI once W and Kuu have been calculated. Note that 
    this function will overcome the memory issues with calculating KSKI conventionally by
    unpacking Kuu... however it will take much longer to compute KSKI. Use this fucntion when 
    unpacking Kuu causes crashing
    
    TO DO : Find another way to do this by significantly reducing runtime.
    '''
    
    N,M = W.shape
    Wt = W.transpose()
    col = np.array([])
    mult = np.zeros((N,1))
    for c in xrange(N):
        mult[c]=1
        
        if c!=0:
            mult[c-1]=0
        w = Wt[:,c].toarray()    
        col = (np.hstack((col,MV_kronprod(K,w))) if col.size else MV_kronprod(K,w))
    return W.dot(col) + (noise**2)*np.eye(N)
