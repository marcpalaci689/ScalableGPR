import numpy as np
import warnings
import math
import CG
from scipy import optimize
import copy
import Kron_utils as KU
import multiprocessing
from multiprocessing import Pool
from contextlib import closing

warnings.filterwarnings("ignore",category=DeprecationWarning)

def RBF(x1,x2,sigma,l): 
    return (sigma**2.0)*np.exp(-(np.sum(x1**2,1).reshape(-1,1) \
            +np.sum(x2**2,1)-2*np.dot(x1,x2.T))/(2.0*l**2))
        
def Gaussian_Kernel(x1,x2,hyp,n=False):
    sigma = math.exp(-hyp[0])
    l = math.exp(-hyp[1])     
    if n:
        s = math.exp(-hyp[2])
        return (sigma**2.0)*np.exp(-(np.sum(x1**2,1).reshape(-1,1)+np.sum(x2**2,1)-2*np.dot(x1,x2.T))/(2.0*l**2)) + (s**2)*np.eye(len(x1))
    else:
        return (sigma**2.0)*np.exp(-(np.sum(x1**2,1).reshape(-1,1) \
                              +np.sum(x2**2,1)-2*np.dot(x1,x2.T))/(2.0*l**2))
 
def ARD(x1,x2,hyp,noise=True):
    N,D = x1.shape
    n   = x2.shape[0]
    sigma = math.exp(-hyp[0])
    s = math.exp(-hyp[-1])
    K = np.zeros((N,n))
    for i in xrange(D):
        x11 = x1[:,i].reshape(-1,1) 
        x22 = x2[:,i].reshape(-1,1)
        l = math.exp(-hyp[i+1])
        K = K - (np.sum(x11**2,1).reshape(-1,1)+np.sum(x22**2,1)-2*np.dot(x11,x22.T))/(2.0*l**2)
    if noise:
        return (sigma**2.0)*np.exp(K) + (s**2)*np.eye(N)
    else:
        return (sigma**2.0)*np.exp(K)
                                         
                                                          
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
    s = math.exp(-hyp[-1])

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
            K.append((sigma**2.0)*np.exp(-(np.sum(xd**2.0,1).reshape(-1,1)+np.sum(xd**2.0,1)-2*np.dot(xd,xd.T))/(2.0*(math.exp(-hyp[d+1]))**2)))  
            E.append(np.real(np.linalg.eigh(K[-1])[0]))
        
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
        complexity = np.sum(np.log(L + (s**2 + rank_fix )*np.ones((N,1))))
    
        # Calculate alpha by Linear CG method
        alpha = CG.Linear_CG(W,K,y,s,rank_fix,tolerance=1e-3,maxiter=2000)
        
        # If the CG does not converge, increase the rank fixing term to give better conditioning
        if alpha[1] != 0:
            print('cg failed, reassigning rank correction term.')
            rank_fix = 100*rank_fix
        else:
            flag = True
            
        # if CG succeeded, return alpha. Else reiterate with a larger rank_fix term.
        if flag:
            break
    alpha = alpha[0]
    
    # Get negative log likelihood (objective function to be minimized)
    return 0.5*(np.dot(y.T,alpha)[0][0] + complexity + N*np.log(2*math.pi)),rank_fix    

def FD_grad((p,P,W,x,y,hyp,rank_fix,epsilon,func)):
        eps = np.zeros((P,1))        
        eps[p]  = epsilon
        f_plus,rank_fix  = Gaussian_Kron(W,x,y,hyp+eps,rank_fix)
        return (f_plus - func)/(epsilon)


def D_Gaussian_Kron(W,x,y,hyp,rank_fix,epsilon=1e-2):
    N,M = W.shape
    P = len(hyp)
    rank_fix = (math.exp(-hyp[0])**2)/1e6 
    # Get negative log likelihood (objective function to be minimized)
    func,rank_fix = Gaussian_Kron(W,x,y,hyp,rank_fix)
      
    
    # Get gradients using centered difference operator
    grad = np.zeros((P,1))
    
    if M>5000:
        l = [(p,P,W,x,y,hyp,rank_fix,epsilon,func) for p in xrange(P)]
        pool = Pool()
        res = pool.imap(FD_grad, l)
        i=0
        for g in res:
            grad[i] = g
            i+=1
        pool.close()
        pool.terminate()
    else:
        for p in xrange(P):
            grad[p] = FD_grad((p,P,W,x,y,hyp,rank_fix,epsilon,func))   
        
    print(func)
    return grad, func, rank_fix


def d_ARD((p,K,L,W,Q,alpha,ind,rank_fix,x,hyp)):
    N,M = W.shape
    D = len(x)
    s = math.exp(-hyp[-1])        
    dK = []
    if p==0:
        gradient = np.dot(alpha.T,KU.kron_MVM(W,K,alpha,0,0))           
        for i in xrange(N):
            j   = ind[i]
            eig = L[i]
            qj = KU.eigenvector(Q,j)
            gradient -= (float(N)/M)*np.dot(qj.T,KU.MV_kronprod(K,qj))/(eig+s**2+rank_fix)
        return gradient
    
    elif p == D+1:
        gradient = np.dot(alpha.T,(s**2)*alpha)
        for i in xrange(N):
            eig = L[i]
            gradient -= (s**2)/(eig + s**2 + rank_fix)
        return gradient                               
    else:
        xd = x[p-1].reshape(-1,1)
        K[p-1] =  (1.0/((math.exp(-hyp[p]))**2))*np.multiply(K[p-1],(np.sum(xd**2,1).reshape(-1,1)\
                                            +np.sum(xd**2,1)-2*np.dot(xd,xd.T)))        
        gradient = 0.5*np.dot(alpha.T,KU.kron_MVM(W,K,alpha,0,0))
        for i in xrange(N):
            j   = ind[i]
            eig = L[i]
            qj = KU.eigenvector(Q,j)
            gradient -= 0.5*(float(N)/M)*np.dot(qj.T,KU.MV_kronprod(K,qj))/(eig + s**2 + rank_fix)
        return gradient       


    
def exact_Gaussian_grad2(W,x,y,hyp,rank_fix):

        
    # Initialize variables    
    N,M = W.shape
    D = len(x)
    P = len(hyp)
    sigma = math.exp(-hyp[0]/D)
    s = math.exp(-hyp[-1])

    # set a flag to know if the CG converged
    flag=False
    
    # iterate 5 times or until CG converged to desired accuracy
    for i in xrange(5):
        # initialize list for dimensional gram matrices, Eigenvalues, Eigenvectors, and gradients
        K  = []
        E  = []
        Q  = []
  
        
        # Calculate and stack K, Q, and E in each dimension.
        for d in xrange(D):
            xd = x[d].reshape(-1,1)
            K.append((sigma**2.0)*np.exp(-(np.sum(xd**2.0,1).reshape(-1,1)+np.sum(xd**2.0,1) \
                                        -2*np.dot(xd,xd.T))/(2.0*(math.exp(-hyp[d+1]))**2)))  
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
            rank_fix = 100*rank_fix
        else:
            flag = True
            
        # if CG succeeded, return alpha. Else reiterate with a larger rank_fix term.
        if flag:
            break
    alpha = alpha[0]
    
    # calculate gradients of Kuu and then the gradient of the likelihood
    grad = np.zeros((P,1))
    
    # If M> 5000 use paralell gradient computation (less than 5000 parallel comunication overcomes benefits)
    if M>5000:
        l = [(i,K,L,W,Q,alpha,ind,rank_fix,x,hyp) for i in xrange(P)]
        pool = Pool()
        res = pool.imap(d_ARD, l)
        i=0
        for g in res:
            grad[i] = g
            i+=1
        pool.close()
        pool.terminate()
    # Else compute gradients sequentially        
    else:
        for p in xrange(P):
            grad[p] = d_ARD((p,K,L,W,Q,alpha,ind,rank_fix,x,hyp))
            
    func = 0.5*(np.dot(y.T,alpha)[0][0] + complexity + N*np.log(2*math.pi))
    print(func)

 
   # Get negative log likelihood (objective function to be minimized)
    return grad,func,rank_fix              

        
class Gaussian:
    
    def __init__(self,model,hyp='auto'):
        
        self.noise = model.noise
        self.D = model.x.shape[1]
        
        # intialize hyperparameters with relatively smooth values           
        self.hyp = -np.ones((model.D+2,1))
        for i in xrange(1,self.D-2):
            self.hyp[i] = -3
        self.hyp[-1] = 4
      
        # if an interpolation matrix W is present, set attribute interpolate to true and set finite differencing epsilon.
        if hasattr(model,'W'):
            self.interpolate = True
            self.epsilon = 1e-2  #for finite differencing
            self.rank_fix = (math.exp(-self.hyp[0])**2)/1e6 # Rank fix term to ensure well conditioned KSKI matrix
        else:
            self.interpolate = False
            self.rank_fix = (math.exp(-self.hyp[0])**2)/1e6 # Rank fix term to ensure well conditioned Gramm matrix
        
                            
    def K(self,x):
        return Gaussian_Kernel(x,x,self.hyp,n=self.noise)
    
    def Ks(self,x,X):
        return ARD(x,X,self.hyp,noise=False)
    
    def Kss(self,X):
        return ARD(X,X,self.hyp,noise=False)
    
    def Kuu(self,x):  
        Kuu = []
        sigma = math.exp(-self.hyp[0]/self.D)
        s =     math.exp(-self.hyp[-1])
        d=1
        for i in x:
            i = i.reshape(-1,1)
            l = math.exp(-self.hyp[d])
            Kuu.append(RBF(i,i,sigma,l))   
            d+=1
        return Kuu
    
    def Grad(self,model,hyp):
        if self.interpolate:
            if self.gradient == 'exact':
                return exact_Gaussian_grad2(model.W,model.grid.x,model.y,hyp,self.rank_fix)
            else:
                return D_Gaussian_Kron(model.W,model.grid.x,model.y,hyp,self.rank_fix,epsilon=self.epsilon)
        else:
            return D_Gaussian(model.x,model.y,hyp,self.rank_fix,n=self.noise)
        

  
                   
    def Optimize(self,model,gradient='exct',maxnumlinesearch=50,random_starts = 2,verbose=False):
        
        self.gradient = gradient                     
        
        print('----------------------------------------------')
        print('          Optimizing Hyperparameters          ')
        print('---------------------------------------------- \n')
        
        print('DETAILS: ')
        print('--> Performing %i non-linear CG(s) with random initialization(s)' %(random_starts+1))
        print('--> Maximum number of line searches =  %i' %(maxnumlinesearch))
        
        if gradient == 'exact':
            print('--> Calculations of gradients: exact gradients')
        else:    
            print('--> Calculation of gradients: Forward Finite-Differencing')
        if model.M>5000:
            print('--> Gradients computed in parallel on %i cores \n' %(multiprocessing.cpu_count()))
        else:
            print('--> Gradients computed on 1 core \n')
        
        # intialize optimum parameters        
        opthyp,optML,i = CG.minimize(self,model,maxnumlinesearch=maxnumlinesearch, maxnumfuneval=None, red=1.0, verbose=verbose) 
        optrankfix = self.rank_fix
        print('CG run #1 done')
        # Discourage very small characteristic lengths        
        #if opthyp[1]>1.5:
        #    optML[-1] = np.inf 
        
        # Rerun optimization with random intializations
        for j in xrange(random_starts):
            self.hyp = 0.1*np.random.randint(-40,10,size=(len(opthyp)-1,1))
            self.hyp = np.vstack((self.hyp,np.random.randint(-2,6)))
            if self.interpolate:
                self.rank_fix = (math.exp(-self.hyp[0])**2)/1e6
            else:
                self.rank_fix = (math.exp(-self.hyp[0])**2)/1e6
            print(self.hyp)
            hyp, ML, i = CG.minimize(self,model,maxnumlinesearch=maxnumlinesearch, maxnumfuneval=None, red=1.0, verbose=verbose) 

            # if new optimum is better than the last, save it.            
            if ML[-1] != -np.inf and ML[-1] <optML[-1]: #and hyp[1]<1.5:
                optML = ML
                opthyp = hyp
                optrankfix = self.rank_fix
            print('--> CG run #%i done:' %(j+2)) 
            print('--> Optimum Marginal Likelihood: %e \n\n' %(ML[-1]))
        
        
        print('**********************************************')
        print('       Global Optimum Hyperparameters:        ')
        print(opthyp)
        print('Global Optimum Marginal Likelihood: ')
        print(optML[-1])
        print('********************************************** \n\n')
        # Save global optimum to the Kernel object
        self.hyp = opthyp
        self.rank_fix = optrankfix 
           
        return  
      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              