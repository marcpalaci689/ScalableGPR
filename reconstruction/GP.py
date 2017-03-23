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


class GPRegression:

	def __init__(self,x,y,noise=False):

		# set training points
		self.x = x
		self.y = y
		self.noise = noise
		# record number of training points
		self.N  = len(self.x)
		return
	
	def SetKernel(self,kernel='Gaussian'):
		if kernel == 'Gaussian':
			self.kernel = Kernels.Gaussian(self,hyp='auto')
	
	def SetHyp(self,hyp):
		self.kernel.hyp = hyp
	
	def OptimizeHyp(self,maxnumlinesearch=50,random_starts=2,verbose=True):	
		self.kernel.Optimize(self,maxnumlinesearch=maxnumlinesearch,random_starts=random_starts,verbose=verbose)
		return
		
	def GPR(self):	
		if math.exp(-self.kernel.hyp[-1])<1e-7:
			K_y = self.kernel.Ks(self.x,self.x) + 1e-7*np.eye(N)
		else:
			K_y = self.kernel.K(self.x)

		success = 0
		try:
			self.L = np.linalg.cholesky(K_y)
		except: 
			iter = -6	
			while not success:
				try:	
					self.L = np.linalg.cholesky(K_y + (10**iter)*(np.eye(N))) 
					success = True						  
				except:
					iter+=1

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
			K_s     = self.kernel.Ks(self.x,X)
			self.mu = np.dot(K_s.T,self.alpha)
		else:
			K_s  = self.kernel.Ks(self.x,X)
			K_ss = self.kernel.Kss(X)
		
			### Calculate posterior mean and covariance 
			### NOTE: we are assuming mean is zero 
			self.mu           = np.dot(K_s.T,self.alpha)
			v                 = np.linalg.solve(self.L,K_s)
			self.variance     = K_ss - np.dot(v.T,v)

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

		return

	def KISSGP(self):
		
		if self.noise:
			noise = self.kernel.hyp[-1]
		else:
			noise = math.log(1e6)
			
		self.Kuu = self.kernel.Kuu(self.grid.x)
		
		self.alpha	 = CG.Linear_CG(self.W,self.Kuu,self.y,math.exp(-noise),tolerance=1e-12)[0]
		
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
	
	N = 50
	
	x1 = np.sort(np.random.normal(scale=2,size=(1,N))).reshape(N,1)
	x2 = np.sort(np.random.normal(scale=2,size=(1,N))).reshape(N,1)
	x1s = np.linspace(-6,6,num=200).reshape(200,1)
	x2s = np.linspace(-6,6,num=200).reshape(200,1)
	x = np.hstack((x1,x2))
	xs=np.hstack((x1s,x2s))
	y= np.sin(x1) + 0.05*np.random.normal(scale=1,size=(N,1))	
	'''
	data = np.load('regression_data.npz')
	x = data['x']
	y = data['y']
	xs = data['xstar']
	'''
	hyp = np.array([[1.0],[1.0],[2.0]])

	'''
	Model  = GPRegression(x,y,noise=True)
	Model.SetKernel('Gaussian')
	Model.SetHyp(hyp)
	Model.OptimizeHyp(random_starts=3)
	Model.GPR()
	Model.Predict(xs)
	
	fig = plt.figure(1)
	plt.clf()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(Model.x[:,0],Model.x[:,1],Model.y,c='r')
	ax.scatter(Model.X[:,0],Model.X[:,1],Model.mu,c = 'g')
	'''

	
	
	
	Model1 = GPRegression(x,y,noise=True)
	Model1.GenerateGrid([20,20])
	Model1.Interpolate()
	Model1.SetKernel('Gaussian')
	Model1.SetHyp(hyp)
	Model1.OptimizeHyp(random_starts=3)
	Model1.KISSGP()
	Model1.Predict(xs)

	
	fig = plt.figure(2)
	plt.clf()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(Model1.x[:,0],Model1.x[:,1],Model1.y,c='r')
	ax.scatter(Model1.X[:,0],Model1.X[:,1],Model1.mu,c='g')	
	
	
	

