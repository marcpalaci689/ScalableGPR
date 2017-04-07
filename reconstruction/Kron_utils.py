import numpy as np
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
 

def kron_MVM2(pre,W,K,y,noise,rank_fix):
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
    b = np.dot(pre,y)
    c = W.dot(MV_kronprod(K,W.transpose().dot(b)))        
    d = (noise**2+rank_fix)*b
    return np.dot(pre,c) + np.dot(pre,d)    


def KSKI_Unpack(W,K,noise):
    '''
     Use this function to unpack KSKI once W and Kuu have been calculated. Note that 
    this function will overcome the memory issues with calculating KSKI conventionally by
    unpacking Kuu... however it will take much longer to compute KSKI. Use this fucntion when 
    unpacking Kuu causes crashing
    
    TO DO : Find another way to do this by significantly reducing runtime.
    '''
    noise = math.exp(-noise)  
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

def kron_MVM(W,K,y,noise,rank_fix):
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
 
    return W.dot(MV_kronprod(K,W.transpose().dot(y))) + (noise**2)*y + rank_fix*y  


def KSKI_Unpack(W,K,noise):
    '''
     Use this function to unpack KSKI once W and Kuu have been calculated. Note that 
    this function will overcome the memory issues with calculating KSKI conventionally by
    unpacking Kuu... however it will take much longer to compute KSKI. Use this fucntion when 
    unpacking Kuu causes crashing
    
    TO DO : Find another way to do this by significantly reducing runtime.
    '''
    noise = math.exp(-noise)  
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
    
    
def largest_eigs(eigs,N,M):
    ''' 
    This function was created to find the N largest eigenvalues of Kuu. If M=N we just unpack
    the eigenvalues of Kuu and return the eigenvalues and their indices. If M>N, then we unpack the
    Kronecker product and perform a partial sort to find the N largest eigenvalues and their 
    corresponding indeces. NOTE that the eigenvalues are NOT returned in order.
    
    Inputs:
        eigs --> List of the unpacked eigenvalues
        N    --> Number of training points
        M    --> Number of inducing points
        
    Outputs:
        kron  --> largest N eigenvalues
        ind   --> corresponding indices of N largest eigenvalues
    '''

    D = len(eigs)
    if N==M:
        kron = np.kron(eigs[-2],eigs[-1])
        for i in xrange(3,D+1):
            kron = np.kron(eigs[-i],kron)
        ind = np.linspace(0,N-1,num=N,dtype=int)
        return kron,ind
    else:
        kron = -np.kron(eigs[-2],eigs[-1])
        for i in xrange(3,D+1):
            kron = np.kron(eigs[-i],kron)
        ind = np.argpartition(kron,N)[:N]
        return -kron[ind],ind
        
def eigenvector(Q,index):
    '''
    This function will return the eigenvector associated with the given index without completely
    unpacking the Kronecker products of the eigenvectors.
    
    INPUTS:
        Q     --> list of all the one-dimensional eigenvector matrices
        index --> The index of the desired eigenvector 
        
    OUTPUT:
        q --> The unpacked eigenvector corresponding to the given index 
    '''
    D = len(Q)
    index = 1.0*index    
    ind=[]
    factor = [1]
    
    for i in Q[1:][::-1]:
        factor.append(factor[-1]*len(i))
    
    ind.append(int(index//factor[-1]))
    index -= ind[-1]*factor[-1]
    for i in xrange(2,D+1):
        ind.append(int(index//factor[-i]))
        index -= ind[-1]*factor[-i]
        
    q = np.kron(Q[-2][:,ind[-2]],Q[-1][:,ind[-1]])
    
    for i in xrange(3,D+1):
        q = np.kron(Q[-i][:,ind[-i]],q)
    return q
        
    




