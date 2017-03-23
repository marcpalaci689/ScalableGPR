import numpy as np
import warnings
import math
import CG
from GP import GPRegression
from scipy import optimize

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
        
def D_Gaussian(x,y,hyp,n=False):    
    N=len(x)
    sigma = math.exp(-hyp[0])
    l = math.exp(-hyp[1]) 
    
    K =  (sigma**2)*np.exp(-(np.sum(x**2,1).reshape(-1,1)+np.sum(x**2,1)\
                             -2*np.dot(x,x.T))/(2.0*l**2))
    dK_dsigma = -2.0*K
    dK_dl     =  -(1.0/l**2)*np.multiply(K,(np.sum(x**2,1).reshape(-1,1)\
                                            +np.sum(x**2,1)-2*np.dot(x,x.T)))    
    
    if n:
        s = math.exp(-hyp[2])
        dK_ds     = -(2*s**2)*np.eye(N)
        if s<1e-7:
            K = K + 1e-7*np.eye(N)
        else:
            K = K + (s**2)*np.eye(N)
        success = 0
        try:
             L = np.linalg.cholesky(K)
        except: 
            iter = -6    
            while not success:
                try:    
                    L = np.linalg.cholesky(K + (10**iter)*(np.eye(N))) 
                    success = True                          
                except:
                    iter+=1
                    
            
        inv_K = np.dot(np.linalg.inv(L).T,np.linalg.inv(L))
        alpha = np.linalg.solve(L.T,np.linalg.solve(L,y))
        d_sigma = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_K),dK_dsigma))
        d_l = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_K),dK_dl))
        d_s = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_K),dK_ds))
        grad = -np.array([[d_sigma],[d_l],[d_s]])
        complexity = sum(2*np.log(np.diag(L)))
        func =  0.5*np.dot(y.T,alpha)+0.5*complexity+0.5*N*np.log(2*math.pi)
        print(func)
        return grad, func  
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
        return grad, func  

def Gaussian_Kron(W,x,y,hyp):
    N,M = W.shape

    D = len(x)
    sigma = math.exp(-hyp[0]/D)
    l = math.exp(-hyp[1]) 
    s = math.exp(-hyp[2])
    
    # initialize list for dimensional gram matrices, Eigenvalues, and Eigenvectors
    K = []
    E = []
    
    # Calculate and stack K, Q, and E in each dimension.
    for d in xrange(D):
        xd = x[d].reshape(-1,1)
        K.append((sigma**2.0)*np.exp(-(np.sum(xd**2.0,1).reshape(-1,1)+np.sum(xd**2.0,1)-2*np.dot(xd,xd.T))/(2.0*l**2)))  
        E.append(np.real(np.linalg.eig(K[-1])[0]))
    
    # Calculate eigenvalues of the inducing points
    L = E[0]
    for d in xrange(1,D):
        L = np.kron(L,E[d])
    
    L = np.sort(L)
    
    # Approximate to eigenvalues of KSKI by a factor M/N    
    L = (float(N)/M)*L.reshape(-1,1)
    
    # Calculate approximate log|KSKI| from L and s    
    complexity = sum(np.log(L[(M-N):]+(s**2)*np.ones((N,1))))

    # Calculate alpha by Linear CG method
    alpha = CG.Linear_CG(W,K,y,s,tolerance=1e-12)[0]

    # Get negative log likelihood (objective function to be minimized)
    return 0.5*(np.dot(y.T,alpha)[0][0] + complexity + N*np.log(2*math.pi))    

def D_Gaussian_Kron(W,x,y,hyp,epsilon=1e-4):
    
    P = len(hyp)
    
    # Get negative log likelihood (objective function to be minimized)
    func = Gaussian_Kron(W,x,y,hyp)

    # Get gradients using centered difference operator
    grad = np.zeros((P,1))
    

    for p in xrange(P):
        # Perturb the parameters and get the centered difference operator
        # Iniialize a perturbation vector 
        eps = np.zeros((P,1))
        eps[p]  = epsilon
        f_plus  = Gaussian_Kron(W,x,y,hyp+eps)
        f_minus = Gaussian_Kron(W,x,y,hyp-eps)
        # record the centered difference operator into the gradient vector
        grad[p] = (f_plus - f_minus)/(2*epsilon)   

    print(func)
    return grad, func
            

        
class Gaussian:
    
    def __init__(self,model,hyp='auto'):
        
        self.noise = model.noise
        self.D = model.x[1]
        
        if hasattr(model,'W'):
            self.interpolate = True
            self.epsilon = 1e-2  #for finite differencing
        else:
            self.interpolate = False
                   
        if hyp=='auto':
            #self.hyp = (np.ones((D+2),1) if self.noise else np.ones((D+1),1))
            if self.interpolate:
                self.hyp=np.array([[5.0],[-5.0],[5.0]])
            else:
                self.hyp = np.array([[1.0],[1.0],[1.0]])
        else:
            self.hyp = hyp
        
        self.x = model.x
        self.y = model.y           
            
    def K(self,x):
        return Gaussian_Kernel(x,x,self.hyp,n=self.noise)
    
    def Ks(self,x,X):
        return Gaussian_Kernel(x,X,self.hyp,n=False)
    
    def Kss(self,X):
        return Gaussian_Kernel(X,X,self.hyp,n=False)
    
    def Kuu(self,x):  
        Kuu = []
        for i in x:
            i = i.reshape(-1,1)
            Kuu.append(Gaussian_Kernel(i,i,self.hyp,n=False))    
        return Kuu
    
    def Grad(self,model,hyp):
        if self.interpolate:
            return D_Gaussian_Kron(model.W,model.grid.x,model.y,hyp,epsilon=self.epsilon)
        else:
            return D_Gaussian(model.x,model.y,hyp,n=self.noise)
        
    '''    
    def f(self,hyp):
        N=len(self.x)            
        complexity = sum(2*np.log(np.diag(self.L)))
        func =  0.5*np.dot(self.y.T,self.alpha)+0.5*complexity+0.5*N*np.log(2*math.pi)
        print(func)
        return func  
        
    def g(self,hyp): 
        hyp = np.ndarray.flatten(hyp)  
        N=len(self.x)
        print(hyp.shape)
        print(hyp)
        sigma = np.exp(-hyp[0])
        l = np.exp(-hyp[1])  
        K =  (sigma**2)*np.exp(-(np.sum(self.x**2,1).reshape(-1,1)+np.sum(self.x**2,1)\
                         -2*np.dot(self.x,self.x.T))/(2.0*l**2))    
        dK_dsigma = -2.0*K
        dK_dl     =  -(1.0/l**2)*np.multiply(K,(np.sum(self.x**2,1).reshape(-1,1)\
                                                 +np.sum(self.x**2,1)-2*np.dot(self.x,self.x.T)))    
                    
        if self.noise:  
            s = np.exp(-hyp[2])
            dK_ds = -(2*s**2)*np.eye(N)
            if s<1e-7:
                K = K + 1e-7*np.eye(N)
            else:
                K = K + (s**2)*np.eye(N)
            success = 0
            try:
                self.L = np.linalg.cholesky(K)
            except: 
                iter = -6    
                while not success:
                    try:    
                        self.L = np.linalg.cholesky(K + (10**iter)*(np.eye(N))) 
                        success = True                          
                    except:
                        iter+=1
                        
            self.alpha = np.linalg.solve(self.L.T,np.linalg.solve(self.L,self.y))         
            inv_K = np.dot(np.linalg.inv(self.L).T,np.linalg.inv(self.L))
            d_sigma = 0.5*np.trace(np.dot((np.dot(self.alpha,self.alpha.T)-inv_K),dK_dsigma))
            d_l = 0.5*np.trace(np.dot((np.dot(self.alpha,self.alpha.T)-inv_K),dK_dl))
            d_s = 0.5*np.trace(np.dot((np.dot(self.alpha,self.alpha.T)-inv_K),dK_ds))
            grad = -np.asarray((d_sigma,d_l,d_s))
            return grad  
        else: 
            self.L = np.linalg.cholesky(K+1e-6*np.eye(N))
            self.alpha = np.linalg.solve(self.L.T,np.linalg.solve(self.L,self.y))   
            inv_K = np.dot(np.linalg.inv(self.L).T,np.linalg.inv(self.L))
            d_sigma = 0.5*np.trace(np.dot((np.dot(self.alpha,self.alpha.T)-inv_K),dK_dsigma))
            d_l = 0.5*np.trace(np.dot((np.dot(self.alpha,self.alpha.T)-inv_K),dK_dl))
            grad = -np.asarray((d_sigma,d_l,d_s))
            return grad 
    '''
                   
    def Optimize(self,model,maxnumlinesearch=50,random_starts = 2,verbose=False):
       
        opthyp,optML,i = CG.minimize(self,model,maxnumlinesearch=maxnumlinesearch, maxnumfuneval=None, red=1.0, verbose=verbose) 
        for i in xrange(random_starts):
            self.hyp = 0.1*np.random.randint(-60,20,size=(len(opthyp)-1,1))
            self.hyp = np.vstack((self.hyp,np.random.randint(1,4)))
            print(self.hyp)
            
            hyp, ML, i = CG.minimize(self,model,maxnumlinesearch=maxnumlinesearch, maxnumfuneval=None, red=1.0, verbose=verbose) 
            if ML[-1] != -np.inf and ML[-1] <optML[-1]:
                optML = ML
                opthyp = hyp
        
        self.hyp = opthyp 
           
        return  
      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              