import numpy as np
import warnings
import math
import CG
from scipy import optimize
import copy
import Kron_utils as KU

warnings.filterwarnings("ignore",category=DeprecationWarning)

def Gaussian_Kernel(x1,x2,hyp,n=False):
    sigma = math.exp(-hyp[0])
    l = math.exp(-hyp[1])     
    if n:
        s = math.exp(-hyp[2])
        return (sigma**2.0)*np.exp(-(np.sum(x1**2,1).reshape(-1,1)+np.sum(x2**2,1)-2*np.dot(x1,x2.T))/(2.0*l**2)) + (s**2)*np.eye(len(x1))
    else:
        return (sigma**2.0)*np.exp(-(np.sum(x1**2,1).reshape(-1,1) \
                              +np.sum(x2**2,1)-2*np.dot(x1,x2.T))/(2.0*l**2))
        
def D_Gaussian(x,y,hyp,rank_fix,n=False):    
    N=len(x)
    sigma = math.exp(-hyp[0])
    l = math.exp(-hyp[1]) 
    flag = False    
    
    rank_fix = (sigma**2)/1e5
    
    K =  (sigma**2)*np.exp(-(np.sum(x**2,1).reshape(-1,1)+np.sum(x**2,1)\
                             -2*np.dot(x,x.T))/(2.0*l**2))
    dK_dsigma = -2.0*K
    dK_dl     =  -(1.0/l**2)*np.multiply(K,(np.sum(x**2,1).reshape(-1,1)\
                                            +np.sum(x**2,1)-2*np.dot(x,x.T)))    
    
    if n:
        s = math.exp(-hyp[2])
        dK_ds     = -(2*s**2)*np.eye(N)
        
        # Loop to perform Cholesky factorisation since Gramm matrix might not be SPD 
        for i in xrange(5):
            try:
                L = np.linalg.cholesky(K+rank_fix*np.eye(N))
                break
            except:
                # Gramm matrix not SPD, increase the rank correction term and reiterate
                flag = True
                #rank_fix = (10**i)*(sigma**2)/1000
                rank_fix = 10*rank_fix
                print('not SPD, reassigning rank corrector')
            # If fixing the rank 4 times still did not work, raise an error.
            if i==4:
                raise ValueError('Gramm matrix not Positive Definite')
        
        #if not flag:
            #rank_fix=0
          
        inv_K = np.dot(np.linalg.inv(L).T,np.linalg.inv(L))
        alpha = np.linalg.solve(L.T,np.linalg.solve(L,y))
        d_sigma = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_K),dK_dsigma))
        d_l = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_K),dK_dl))
        d_s = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_K),dK_ds))
        grad = -np.array([[d_sigma],[d_l],[d_s]])
        complexity = sum(2*np.log(np.diag(L)))
        func =  0.5*np.dot(y.T,alpha)+0.5*complexity+0.5*N*np.log(2*math.pi)
        print(func)
        return grad, func, rank_fix  
    else:    

        L = np.linalg.cholesky(K+1e-6*np.eye(N))
        inv_K = np.dot(np.linalg.inv(L).T,np.linalg.inv(L))
        alpha = np.linalg.solve(L.T,np.linalg.solve(L,y))
        d_sigma = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_K),dK_dsigma))
        d_l = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_K),dK_dl))
        grad = -np.array([[d_sigma],[d_l],[d_s]])
        complexity = sum(2*np.log(np.diag(L)))
        func =  0.5*np.dot(y.T,alpha)+0.5*complexity+0.5*N*np.log(2*math.pi)
        print(func)
        return grad, func, rank_fix  

def Gaussian_Kron(W,x,y,hyp,rank_fix):
    # Initialize variables    
    N,M = W.shape
    D = len(x)
    sigma = math.exp(-hyp[0]/D)
    l = math.exp(-hyp[1]) 
    s = math.exp(-hyp[2])

    # set a flag to know if the CG converged
    flag=False
    
    # iterate 5 times or until CG converged to desired accuracy
    for i in xrange(5):
        # initialize list for dimensional gram matrices, Eigenvalues, and Eigenvectors
        K = []
        E = []
        
        # Calculate and stack K, Q, and E in each dimension.
        for d in xrange(D):
            xd = x[d].reshape(-1,1)
            K.append((sigma**2.0)*np.exp(-(np.sum(xd**2.0,1).reshape(-1,1)+np.sum(xd**2.0,1)-2*np.dot(xd,xd.T))/(2.0*l**2)))  
            E.append(np.real(np.linalg.eig(K[-1])[0]))
        
        # Calculate eigenvalues of the inducing points
        '''
        L = E[0]
        for d in xrange(1,D):
            L = np.kron(L,E[d])
        
        L = np.sort(L,kind='mergesort')
        '''
        L,ind = KU.largest_eigs(E,N,M)
        
        # Approximate to eigenvalues of KSKI by a factor M/N    
        L = (float(N)/M)*L
        
        # Calculate approximate log|KSKI| from L and s    
        complexity = np.sum(np.log(L + (s**2 + rank_fix)*np.ones((N,1))))
    
        # Calculate alpha by Linear CG method
        alpha = CG.Linear_CG(W,K,y,s,rank_fix,tolerance=1e-3,maxiter=2000)
        
        # If the CG does not converge, increase the rank fixing term to give better conditioning
        if alpha[1] != 0:
            print('cg failed, reassigning rank correction term.')
            #rank_fix = (10**i)*(sigma**2)/100
            rank_fix = 100*rank_fix
        else:
            flag = True
            
        # if CG succeeded, return alpha. Else reiterate with a larger rank_fix term.
        if flag:
            break
    alpha = alpha[0]
    
    # Get negative log likelihood (objective function to be minimized)
    return 0.5*(np.dot(y.T,alpha)[0][0] + complexity + N*np.log(2*math.pi)),rank_fix    

def D_Gaussian_Kron(W,x,y,hyp,rank_fix,epsilon=1e-2,finitedifferencing = 'forward'):
    
    P = len(hyp)
    rank_fix = (math.exp(-hyp[0])**2)/1e5 
    # Get negative log likelihood (objective function to be minimized)
    func,rank_fix = Gaussian_Kron(W,x,y,hyp,rank_fix)
      
    
    # Get gradients using centered difference operator
    grad = np.zeros((P,1))
    
    if finitedifferencing == 'centered':
        for p in xrange(P):
            # Perturb the parameters and get the centered difference operator
            # Iniialize a perturbation vector 
            eps = np.zeros((P,1))
            eps[p]  = epsilon
            f_plus,rank_fix  = Gaussian_Kron(W,x,y,hyp+eps,rank_fix)
            f_minus,rank_fix = Gaussian_Kron(W,x,y,hyp-eps,rank_fix)
            # record the centered difference operator into the gradient vector
            grad[p] = (f_plus - f_minus)/(2*epsilon)   
    else:
        for p in xrange(P):
            # Perturb the parameters and get the centered difference operator
            # Iniialize a perturbation vector 
            eps = np.zeros((P,1))
            eps[p]  = epsilon
            f_plus,rank_fix  = Gaussian_Kron(W,x,y,hyp+eps,rank_fix)
            # record the centered difference operator into the gradient vector
            grad[p] = (f_plus - func)/(epsilon)           
    print(func)
    return grad, func, rank_fix


def exact_Gaussian_grad(W,x,y,hyp,rank_fix):

        
    # Initialize variables    
    N,M = W.shape
    D = len(x)
    sigma = math.exp(-hyp[0]/D)
    l = math.exp(-hyp[1]) 
    s = math.exp(-hyp[2])

    # set a flag to know if the CG converged
    flag=False
    
    # iterate 5 times or until CG converged to desired accuracy
    for i in xrange(5):
        # initialize list for dimensional gram matrices, Eigenvalues, Eigenvectors, and gradients
        K  = []
        E  = []
        Q  = []
        dK = []
        
        # Calculate and stack K, Q, and E in each dimension.
        for d in xrange(D):
            xd = x[d].reshape(-1,1)
            K.append((sigma**2.0)*np.exp(-(np.sum(xd**2.0,1).reshape(-1,1)+np.sum(xd**2.0,1)-2*np.dot(xd,xd.T))/(2.0*l**2)))  
            e,q = np.linalg.eigh(K[-1])
            E.append(e)
            Q.append(q)
        
        # get N largest eigenvalues
        L,ind = KU.largest_eigs(E,N,M)
        
              
        
        # Approximate to eigenvalues of KSKI by a factor M/N    
        L = (float(N)/M)*L
        
        # Calculate approximate log|KSKI| from L and s    
        complexity = np.sum(np.log(L + (s**2 + rank_fix)*np.ones((N,1))))
    
        # Calculate alpha by Linear CG method
        alpha = CG.Linear_CG(W,K,y,s,rank_fix,tolerance=1e-3,maxiter=2000)
        
        # If the CG does not converge, increase the rank fixing term to give better conditioning
        if alpha[1] != 0:
            print('cg failed, reassigning rank correction term.')
            #rank_fix = (10**i)*(sigma**2)/100
            rank_fix = 10*rank_fix
        else:
            flag = True
            
        # if CG succeeded, return alpha. Else reiterate with a larger rank_fix term.
        if flag:
            break
    alpha = alpha[0]
    
    # calculate gradients of Kuu and then the gradient of the likelihood
    grad = np.zeros((P,1))

    for p in xrange(len(hyp)):      
        if p==0:
            for d in xrange(D):
                dK.append((-2/D)*K[d])
            grad = -np.dot(alpha.T,KU.kron_MVM(W,dK,alpha,0,0))
            # have to figure out which eigenvectors to use                
            for i in xrange(N):
                j   = ind[i]
                eig = L[i]
                qj = KU.eigenvector(Q,j)
                grad += np.dot(qj.T,KU.MVM_kronprod(dK,qj))/(eig+s**2 + rank_fix)
            grad[p]= grad
        elif p==1:
            for d in xrange(D):
                dk.append(-(1.0/l**2)*np.multiply(K[d],(np.sum(x[d]**2,1).reshape(-1,1)\
                                            +np.sum(x[d]**2,1)-2*np.dot(x[d],x[d].T)))
            grad = -np.dot(alpha.T,KU.kron_MVM(W,dK,alpha,0,0))
            for i in xrange(N):
                j   = ind[i]
                eig = L[i]
                qj = KU.eigenvector(Q,j)
                grad += np.dot(qj.T,KU.MVM_kronprod(dK,qj))/(eig + s**2 + rank_fix)
            grad[p]= grad   
        else:
            dK = -(2*s**2)*np.eye(N)                             
            grad = -np.dot(alpha.T,np.dot(dK,alpha))
            for i in xrange(N):
                eig = L[i]
                grad += (-2*s**2)/(eig + s**2 + rank_fix)
            grad[p]= grad                           

            
    func = 0.5*(np.dot(y.T,alpha)[0][0] + complexity + N*np.log(2*math.pi)),rank_fix
    # Get negative log likelihood (objective function to be minimized)
    return grad,func,rank_fix              
'''
        
class Gaussian:
    
    def __init__(self,model,hyp='auto'):
        
        self.noise = model.noise
        self.D = model.x.shape[1]
        
        # intialize hyperparameters with relatively smooth values           
        if hyp=='auto':
            #self.hyp = (np.ones((D+2),1) if self.noise else np.ones((D+1),1))
            self.hyp = np.array([[-1.0],[-3.0],[6.0]])
        else:
            self.hyp = hyp
        
        # if an interpolation matrix W is present, set attribute interpolate to true and set finite differencing epsilon.
        if hasattr(model,'W'):
            self.interpolate = True
            self.epsilon = 1e-2  #for finite differencing
            self.rank_fix = (math.exp(-self.hyp[0])**2)/1e5 # Rank fix term to ensure well conditioned KSKI matrix
        else:
            self.interpolate = False
            self.rank_fix = (math.exp(-self.hyp[0])**2)/1e2 # Rank fix term to ensure well conditioned Gramm matrix
        
                  
            
    def K(self,x):
        return Gaussian_Kernel(x,x,self.hyp,n=self.noise)
    
    def Ks(self,x,X):
        return Gaussian_Kernel(x,X,self.hyp,n=False)
    
    def Kss(self,X):
        return Gaussian_Kernel(X,X,self.hyp,n=False)
    
    def Kuu(self,x):  
        Kuu = []
        hyper = copy.deepcopy(self.hyp) 
        hyper[0][0] = float(hyper[0][0])/self.D
        for i in x:
            i = i.reshape(-1,1)
            Kuu.append(Gaussian_Kernel(i,i,hyper,n=False))    
        return Kuu
    
    def Grad(self,model,hyp):
        if self.interpolate:
            return D_Gaussian_Kron(model.W,model.grid.x,model.y,hyp,self.rank_fix,epsilon=self.epsilon)
        else:
            return D_Gaussian(model.x,model.y,hyp,self.rank_fix,n=self.noise)
        

  
                   
    def Optimize(self,model,maxnumlinesearch=50,random_starts = 2,verbose=False):
        
        # intialize optimum parameters        
        opthyp,optML,i = CG.minimize(self,model,maxnumlinesearch=maxnumlinesearch, maxnumfuneval=None, red=1.0, verbose=verbose) 
        optrankfix = self.rank_fix
        
        # Discourage very small characteristic lengths        
        if opthyp[1]>1.5:
            optML[-1] = np.inf 
        
        # Rerun optimization with random intializations
        for i in xrange(random_starts):
            self.hyp = 0.1*np.random.randint(-50,10,size=(len(opthyp)-1,1))
            self.hyp = np.vstack((self.hyp,np.random.randint(-2,6)))
            if self.interpolate:
                self.rank_fix = (math.exp(-self.hyp[0])**2)/1e5
            else:
                self.rank_fix = (math.exp(-self.hyp[0])**2)/1e5
            print(self.hyp)
            hyp, ML, i = CG.minimize(self,model,maxnumlinesearch=maxnumlinesearch, maxnumfuneval=None, red=1.0, verbose=verbose) 

            # if new optimum is better than the last, save it.            
            if ML[-1] != -np.inf and ML[-1] <optML[-1] and hyp[1]<1.5:
                optML = ML
                opthyp = hyp
                optrankfix = self.rank_fix
        print('Optimum Hyperparameters: ')
        print(opthyp)
        print('Marginal Likelihood: ')
        print(optML[-1])
        
        # Save global optimum to the Kernel object
        self.hyp = opthyp
        self.rank_fix = optrankfix 
           
        return  
      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              