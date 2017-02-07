import numpy as np
import matplotlib.pyplot as plt
import math
import time
import solver 
import gc
import sklearn.datasets
import sample
import pyGPs
import Kronecker as kron
from kernels import Gaussian

class GPRegression:

	def __init__(self,parameters):

		### Load dataset
		if parameters['dataset'] == 'Classic':
			data   = np.load('2d_data.npz')
			self.x = data['x']
			self.y = data['y']
			self.X = data['xstar']


		elif parameters['dataset'] == 'Modified':
			self.x = parameters['x']
			self.y = parameters['y']
			self.X = parameters['X']

		else:
			raise ValueError("Invalid dataset")

		### Initialize Hyperparameters
		self.N      = len(self.x)
		self.n      = len(self.X)

		self.parameters = parameters
		self.s      = parameters['s']
		self.sigma  = parameters['sigma'] 
		self.l      = parameters['l']
		self.params = np.array([[self.sigma],[self.l],[np.exp(self.s)]])

	


	def GP_Regression(self):
	
		self.params = np.array([[self.sigma],[self.l],[self.s]])
		
		K_y   = Gaussian(self.x,self.x,self.sigma,self.l)+self.s*np.eye(self.N)
		K_s   = Gaussian(self.x,self.X,self.sigma,self.l)
		K_ss = Gaussian(self.X,self.X,self.sigma,self.l)

		### Calculate posterior mean and covariance 
		### NOTE: we are assuming mean is zero 
		L            = np.linalg.cholesky(K_y)
		self.alpha   = np.linalg.solve(L.T,np.linalg.solve(L,self.y))
		self.mu_s    = np.dot(K_s.T,self.alpha)

		v            = np.linalg.solve(L,K_s)
		self.sigma_s = K_ss - np.dot(v.T,v)

		### get pointwise standard deviation for plotting purposes
		self.s_s     = np.sqrt(np.diag(self.sigma_s)).reshape(-1,1)

		### Calculate marginal likelihood
		self.marginal_likelihood = 0.5*np.dot(self.y.T,self.alpha) + \
		0.5*np.log(np.linalg.det(K_y)) +0.5*self.N*np.log(2*math.pi)
		
		return 


	def KISS_GP(self):
		
		# Initialize grid
		self.grid 	 = kron.tensor_grid(self.x,[5000,5000])
		
		# Generate grid
		self.grid.generate(self.parameters)	
		
		#Perform Interpolation
		self.grid.SKI()
		
		self.alpha	 = solver.CG(self.grid.W,self.grid.Kd,self.y,sigma=self.s,tolerance=1e-7)
		
		K_s   		 = Gaussian(self.x,self.X,self.sigma,self.l)
		
		self.mu_s    = np.dot(K_s.T,self.alpha[0])
		
	
		
		return 

	def PlotModel(self):
		### plots
		plt.figure(1)
		plt.clf()
		
		plt.plot(self.x,self.y,'ro', label='Training points')
		plt.plot(self.X,self.mu_s,'g', label='GP average value')
		plt.fill_between(self.X.flatten(), (self.mu_s-2*self.s_s).flatten(), (self.mu_s+2*self.s_s).flatten(),color='blue',alpha=0.25,interpolate=True)

		plt.legend(loc=0)
		plt.title('Mean predictions plus 3 st.deviations')
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.axis([np.min(self.X),np.max(self.X),np.min(self.mu_s-2*self.s_s),np.max(self.mu_s+2*self.s_s)])
		plt.show()





if __name__ == '__main__':
	gc.collect()
	parameters={ 'dataset':'Classic',
				 'n':100,
		         's':math.exp(-4),
		         'sigma':math.exp(0.5),
		         'l':math.exp(3),
		         'kernel': 'SE'
		         }
	'''
	demoData = np.load('regression_data.npz')
	x = demoData['x']
	y = demoData['y']
	z = demoData['xstar']
	
	
	model = pyGPs.GPR()
	model.getPosterior(x,y)
	model.optimize(x,y)
	model.predict(z)
	model.plot()
	'''
	
	sample = GPRegression(parameters)
	sample.KISS_GP()
	
	sample1 = GPRegression(parameters)
	sample1.GP_Regression()
	'''
	k = np.kron(sample.grid.Kd[0],sample.grid.Kd[1])
	W = sample.grid.W.toarray()
	
	kski = np.dot(W,np.dot(k,W.T))
	
	Ky = Gaussian(sample1.x,sample1.x,parameters['sigma'],parameters['l'])
	'''
#sample.KISS_GP()
#ax.fill_between(self.X.T, (self.mu_s-2*self.s_s).T, (self.mu_s+2*self.s_s).T,color='blue',interpolate=True)

		
		

