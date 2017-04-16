import numpy as np
import matplotlib.pyplot as plt
import math
import time
import gc
import Kernels 
import Grid
import CG
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Kron_utils as KU


class GPRegression:

    def __init__(self,x,y,noise=False):

        # set training points
        self.x = x
        self.y = y
        self.D = x.shape[1]
        self.noise = noise
        # record number of training points
        self.N  = len(self.x)
        return
    
    def SetKernel(self,kernel='Gaussian'):
        if kernel == 'Gaussian':
            self.kernel = Kernels.Gaussian(self,self.D)      
    
    def SetHyp(self,hyp):
        self.kernel.hyp = hyp
        if self.kernel.__class__.__name__ == 'Gaussian':
            if self.kernel.interpolate:
                self.kernel.rank_fix = (math.exp(-hyp[0])**2)/1e6
            else:
                self.kernel.rank_fix = (math.exp(-hyp[0])**2)/1e6
                
    def OptimizeHyp(self,maxnumlinesearch=50,random_starts=2,verbose=True):    
        self.kernel.Optimize(self,maxnumlinesearch=maxnumlinesearch,random_starts=random_starts,verbose=verbose)
        return
        
    def GPR(self):    
        self.Ky = self.kernel.Ks(self.x,self.x)
        
        self.L = np.linalg.cholesky(self.Ky+self.kernel.rank_fix*np.eye(self.N))    
        ### Calculate posterior mean and covariance 
        ### NOTE: we are assuming mean is zero 
    
        self.alpha   = np.linalg.solve(self.L.T,np.linalg.solve(self.L,self.y))

        ### Calculate marginal likelihood
        complexity = sum(2*np.log(np.diag(self.L)))
        self.ML =  0.5*np.dot(self.y.T,self.alpha) \
                    +0.5*complexity+0.5*self.N*np.log(2*math.pi)        
        return 
    
    def Predict(self,X):
        self.X = X
        if self.kernel.interpolate:
            self.K_s     = self.kernel.Ks(self.x,self.X)
            self.mu = np.dot(self.K_s.T,self.alpha)
        else:
            self.K_s  = self.kernel.Ks(self.x,self.X)
            self.K_ss = self.kernel.Kss(X)
        
            ### Calculate posterior mean and covariance 
            ### NOTE: we are assuming mean is zero 
            self.mu           = np.dot(self.K_s.T,self.alpha)
            v                 = np.linalg.solve(self.L,self.K_s)
            self.variance     = self.K_ss - np.dot(v.T,v)

        ### get pointwise standard deviation for plotting purposes
        #self.var = np.sqrt(np.diag(self.variance)).reshape(-1,1)
    
        return 
    
    def GenerateGrid(self,grid):
        # Initialize grid
        self.grid = Grid.Grid(self.x,grid)
        
        return
    
    def Interpolate(self,scheme='cubic'):
        # Interpolate grid to find Weight Matrix
        self.W = self.grid.Interpolate(self.x,scheme=scheme)
        self.M = self.W.shape[1]

        return

    def KISSGP(self):
        
        if self.noise:
            noise = self.kernel.hyp[-1]
        else:
            noise = math.log(1e6)            
        self.Kuu = self.kernel.Kuu(self.grid.x)
        
        alpha     = CG.Linear_CG(self.W,self.Kuu,self.y,math.exp(-noise),self.kernel.rank_fix,tolerance=1e-5,maxiter=5000)
        print(alpha[1])
        self.alpha     = alpha[0]
        return 

    def PlotModel(self):
        ### plots
        plt.figure(1)
        plt.clf()
        
        plt.plot(self.x,self.y,'ro', label='Training points')
        plt.plot(self.X,self.mu,'g', label='GP average value')
        plt.fill_between(self.X.flatten(), (self.mu-2*self.var).flatten(), (self.mu+2*self.var).flatten(),color='blue',alpha=0.25,interpolate=True)

        plt.legend(loc=0)
        plt.title('Mean predictions plus 3 st.deviations')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis([np.min(self.X),np.max(self.X),np.min(self.mu-2*self.var),np.max(self.mu+2*self.var)])
        plt.show()





if __name__ == '__main__':
    
    gc.collect()
    
    N = 1500
    '''
    x1 = np.sort(np.random.normal(scale=10,size=(1,N))).reshape(N,1)
    x2 = np.sort(np.random.normal(scale=10,size=(1,N))).reshape(N,1)
    '''
    x1 = np.sort(25*np.random.rand(1,N)-25*np.random.rand(1,N)).reshape(N,1)
    x2 =  np.sort(25*np.random.rand(1,N)-25*np.random.rand(1,N)).reshape(N,1)
    x1s = np.linspace(-28,28,num=300).reshape(300,1)
    x2s = np.linspace(-28,28,num=300).reshape(300,1)
    x = np.hstack((x1,x2))
    xs=np.hstack((x1s,x2s))
    y=  100*(np.sin(x2))   #x1**2 - 10*x1*(np.sin(x2))**3  + np.random.normal(scale=10,size=(N,1)) - 10*x1*(np.sin(x2))**3 
    hyp = np.array([[2.81609236],[-4.81052742],[-0.56815406],[4.27924204]])

    '''
    Model  = GPRegression(x,y,noise=True)
    Model.SetKernel('Gaussian')
    Model.SetHyp(hyp)
    start = time.time()
    Model.OptimizeHyp(maxnumlinesearch=25,random_starts=8)
    end = time.time()
    Model.GPR()
    Model.Predict(xs)
    
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Model.x[:,0],Model.x[:,1],Model.y,c='r')
    ax.scatter(Model.X[:,0],Model.X[:,1],Model.mu,c = 'g')
    
    
    print('Standard GP done in %.8f seconds' %(end-start))
    '''    
    
    
    Model1 = GPRegression(x,y,noise=True)
    Model1.GenerateGrid([40,40])
    Model1.Interpolate(scheme='cubic')
    Model1.SetKernel('Gaussian')
    Model1.SetHyp(hyp)
    start = time.time()
    #Model1.OptimizeHyp(maxnumlinesearch=40,random_starts=3)
    end = time.time()
    Model1.KISSGP()
    Model1.Predict(xs)
    
    print('Kiss-GP done in %.8f seconds' %(end-start))
    
    
    fig = plt.figure(2)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Model1.x[:,0],Model1.x[:,1],Model1.y,c='r')
    ax.scatter(Model1.X[:,0],Model1.X[:,1],Model1.mu,c='g')    
    
    '''
    print(np.linalg.norm(Model1.mu-Model.mu))
    
    kski = KU.KSKI_Unpack(Model1.W,Model1.Kuu,Model1.kernel.hyp[2])
    
    print(np.linalg.norm(Model.Ky-kski))
    
    print(np.linalg.norm(Model.alpha-Model1.alpha))
    '''
    
    '''
    with open('gp.npz','wb') as f1:
        np.savez(f1,x=Model.mu,y=Model.x,z=Model.y,w=Model.X)

    with open('kissgp.npz','wb') as f2:
        np.savez(f2,x=Model1.mu,y=Model1.x,z=Model1.y,w=Model1.X)
    '''

    start = time.time()
    g1,f1,r1 = Kernels.exact_Gaussian_grad2(Model1.W,Model1.grid.x,Model1.y,Model1.kernel.hyp,Model1.kernel.rank_fix)
    end=time.time()   
    print(end-start)
    print(' ')    
   
    
    start=time.time()
    g2,f2,r2 = Kernels.D_Gaussian_Kron(Model1.W,Model1.grid.x,Model1.y,Model1.kernel.hyp,Model1.kernel.rank_fix,epsilon=1e-2)
    end = time.time()
    print(end-start)
    print(' ') 
  
    
    