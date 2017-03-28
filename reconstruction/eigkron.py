import numpy as np 
import time
import GP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gp_mu = []
kissgp_mu = []

gp_hyp = []
kissgp_hyp = []

gp_time = []
kissgp_time=[]

target = []

trainx = []
trainy = []
testX = []

#N = [1024,2500,4900]
n = [625,1024,2500,4900,6400]


for N in n:

    # Generate a 2d synthetic dataset
    m = int(N**0.5)    
    x1 = np.sort(25*np.random.rand(1,N)-25*np.random.rand(1,N)).reshape(N,1)
    x2 =  np.sort(25*np.random.rand(1,N)-25*np.random.rand(1,N)).reshape(N,1)
    x1s = np.linspace(-24,24,num=300).reshape(300,1)
    x2s = np.linspace(-24,24,num=300).reshape(300,1)
    x = np.hstack((x1,x2))
    xs=np.hstack((x1s,x2s))
    y= x1**2 - 10*x1*(np.sin(x2))**3 + np.random.normal(scale=10,size=(N,1))
    ys = x1s**2 - 10*x1s*(np.sin(x2s))**3 
    
    
    hyp=np.array([[-3.0],[-4.0],[6.0]])
    
    # Perform standard Guassian Process	
    GPR  = GP.GPRegression(x,y,noise=True)
    GPR.SetKernel('Gaussian')
    GPR.SetHyp(hyp)
    start = time.time()
    GPR.OptimizeHyp(maxnumlinesearch=20,random_starts=4)
    end = time.time()
    print('Standard GP done in %.8f seconds' %(end-start))
    gp_opttime = end-start
    GPR.GPR()
    GPR.Predict(xs)
    
    
    # Perform a Structured Kernel Interpolation regression 	
    KISSGP = GP.GPRegression(x,y,noise=True)
    KISSGP.GenerateGrid([m,m])
    KISSGP.Interpolate(scheme='cubic')
    KISSGP.SetKernel('Gaussian')
    KISSGP.SetHyp(hyp)
    start = time.time()
    KISSGP.OptimizeHyp(maxnumlinesearch=20,random_starts=4)
    end = time.time()
    print('Kiss-GP done in %.8f seconds' %(end-start))
    ski_opttime = end-start
    KISSGP.KISSGP()
    KISSGP.Predict(xs)

    gp_mu.append(GPR.mu)
    kissgp_mu.append(KISSGP.mu)
    
    gp_hyp.append(GPR.kernel.hyp)
    kissgp_hyp.append(KISSGP.kernel.hyp)
    
    gp_time.append(gp_opttime)
    kissgp_time.append(ski_opttime)
    
    target.append(ys)
    
    trainx.append(x)
    trainy.append(y)
    testX.append(xs)
'''

# Plot training points and predictive curve on 3d plot for GP
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(GPR.x[:,0],GPR.x[:,1],GPR.y,c='r')
ax.scatter(GPR.X[:,0],GPR.X[:,1],GPR.mu,c = 'g')


# Plot training points and predictive curve on 3d plot for KISSGP	
fig = plt.figure(2)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(KISSGP.x[:,0],KISSGP.x[:,1],KISSGP.y,c='r')
ax.scatter(KISSGP.X[:,0],KISSGP.X[:,1],KISSGP.mu,c='g')	


plt.show()
	

with open('gp.npz','wb') as f1:
	np.savez(f1,x=Model.mu,y=Model.x,z=Model.y,w=Model.X)

with open('kissgp.npz','wb') as f2:
	np.savez(f2,x=Model1.mu,y=Model1.x,z=Model1.y,w=Model1.X)
'''
	
'''
KISSGP1 = GP.GPRegression(x,y,noise=True)
KISSGP1.GenerateGrid([80,80])
KISSGP1.Interpolate(scheme='cubic')
KISSGP1.SetKernel('Gaussian')
KISSGP1.SetHyp(KISSGP.kernel.hyp)
KISSGP1.kernel.rank_fix = KISSGP.kernel.rank_fix
KISSGP1.KISSGP()
KISSGP1.Predict(xs)
    
fig = plt.figure(3)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(KISSGP.X[:,0],KISSGP.X[:,1],ys,c='r')
ax.scatter(KISSGP.X[:,0],KISSGP.X[:,1],KISSGP.mu,c='g')	
 '''   
  
gp_mu=np.array(gp_mu)
kissgp_mu=np.array(kissgp_mu)

gp_hyp = np.array(gp_hyp)
kissgp_hyp = np.array(kissgp_hyp)

gp_time = np.array(gp_time)
kissgp_time = np.array(kissgp_time)

target = np.array(target)

trainx = np.array(trainx)
trainyy = np.array(trainy)  
testX = np.array(testX)
n = np.array(n)

with open('gp.npz','wb') as f1:
	np.savez(f1,x=trainx,y=trainy,X=testX,target=target,mu=gp_mu,hyp=gp_hyp,time=gp_time,N=n)
with open('kissgp.npz','wb') as f2:
	np.savez(f2,x=trainx,y=trainy,X=testX,target=target,mu=kissgp_mu,hyp=kissgp_hyp,time=kissgp_time,N=n)