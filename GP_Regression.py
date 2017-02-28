import numpy as np
import matplotlib.pyplot as plt
import math
import time
import solver 
import gc
import sample
import pyGPs
import Kronecker as kron
from kernels import Gaussian
from kernels import NoNoise_Gaussian
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import kernels 

class GPRegression:

	def __init__(self,parameters):

		### Load dataset
		if parameters['dataset'] == 'Classic':
			data   = np.load('regression_data.npz')
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

	
	def optimizeHyperparameters(self,kron=True):
		
		if kron == False:
			K_y = Gaussian(self.x,self.x,self.sigma,self.l,self.s)
			### Calculate posterior mean and covariance 
			### NOTE: we are assuming mean is zero 
			L            = np.linalg.cholesky(K_y)
			self.alpha   = np.linalg.solve(L.T,np.linalg.solve(L,self.y))
			min = 0.5*np.dot(self.y.T,self.alpha) + \
			0.5*np.log(np.linalg.det(K_y)) +0.5*self.N*np.log(2*math.pi)
			if min==-np.inf: min=np.inf
			print (min)
			params = np.array([[self.sigma],[self.l],[self.s]])
			best_params = params
		
			
			params,ML,i = solver.minimize(params,self.x,self.y)
			print(len(ML))
			print(ML[-1])
			if ML[-1] != -np.inf and ML[-1] < min:
				min = ML[-1]
				best_params = params		
			
			iter= 1
			while iter<10:
				params = 0.2*np.random.randint(-20,20,size=(3,1))
				params,ML,i = solver.minimize(params,self.x,self.y)
				print(len(ML))
				print(ML[-1])
				if ML[-1] != -np.inf and ML[-1] < min:
					min = ML[-1]
					best_params = params
				iter+=1
				
			
			self.s = best_params[2]
			self.sigma = best_params[0]
			self.l = best_params[1]
			self.parameters['s'] = self.s
			self.parameters['sigma'] = self.sigma
			self.parameters['l'] = self.l
		
		if kron == True:
			
			params = np.array([[self.sigma],[self.l],[self.s]])
			min = kernels.Gaussian_Kron(self.grid.W,self.grid.dims,self.y,params)
			if min==-np.inf: min=np.inf
			best_params = params
			print(min)
			
			params,ML,i = solver.minimize_kron(params,self.grid.W,self.grid.dims,self.y)
			print(ML[-1])
			if ML[-1] != -np.inf and ML[-1] < min :
				print(len(ML))
				min = ML[-1]
				best_params = params		
			
			iter= 1
			while iter<10:
				params = 0.2*np.random.randint(-20,20,size=(3,1))
				params,ML,i = solver.minimize_kron(params,self.grid.W,self.grid.dims,self.y)
				print(len(ML))
				print(ML[-1])
				if ML[-1] != -np.inf and ML[-1] < min:
					min = ML[-1]
					best_params = params
					print(best_params)
				iter+=1
				
			
			self.s = best_params[2]
			self.sigma = best_params[0]
			self.l = best_params[1]
			self.parameters['s'] = self.s
			self.parameters['sigma'] = self.sigma
			self.parameters['l'] = self.l				
		
		
	def GP_Regression(self):
	
		
		K_y = Gaussian(self.x,self.x,self.sigma,self.l,self.s)
		K_s   = NoNoise_Gaussian(self.x,self.X,self.sigma,self.l)
		K_ss = NoNoise_Gaussian(self.X,self.X,self.sigma,self.l)
		self.K_s = K_s
	
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
	
	def Inducing_Points(self):
		# Initialize grid
		self.grid 	 = kron.tensor_grid(self.x,[125,125])
		
		# Generate grid
		self.grid.generate(self.parameters)	
		
		#Perform Interpolation
		self.grid.SKI(interpolation='cubic')
		
		return


	def KISS_GP(self):
		self.grid.generate(self.parameters)	
		self.alpha	 = solver.Linear_CG(self.grid.W,self.grid.Kd,self.y,self.s,tolerance=1e-12)
		
		K_s   		 = NoNoise_Gaussian(self.x,self.X,self.sigma,self.l)
		self.K_s = K_s
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
	
	data = np.load('test_data.npz')
	x1 = np.linspace(0,50,num=1000).reshape(1000,1)
	x2 = np.linspace(0,50,num=1000).reshape(1000,1)
	x1s = np.linspace(-1,51,num=500).reshape(500,1)
	x2s = np.linspace(-1,51,num=500).reshape(500,1)
	x = np.hstack((x1,x2))
	xs=np.hstack((x1s,x2s))
	y= 10*np.sin(0.1*(x1+x2)).reshape(1000,1)
	parameters={ 'dataset':'Modified',
				 'n':1000,
		         's': 1,
		         'sigma': -1,
		         'l': 1 ,
		         'kernel': 'SE',
		         'x' : x,
		         'y' : y,
		         'X' : xs,
		         }
	
	'''
	sample = GPRegression(parameters)
	#sample.optimizeHyperparameters(kron=False)
	sample.GP_Regression()
	'''
	
	
	sample1 = GPRegression(parameters)
	sample1.Inducing_Points()
	sample1.optimizeHyperparameters()
	sample1.KISS_GP() 
	
	
	#K    = np.kron(sample1.grid.Kd[0],sample1.grid.Kd[1])
	#Kski = sample1.grid.W.dot((sample1.grid.W.dot(K)).T).T + ((math.exp(-sample1.s))**2)*np.eye(N)
	#sample.GP_Regression()

	'''
	hyp = np.array([[parameters['sigma']],[parameters['l']],[parameters['s']]])
	
	g,f,L = kernels.Derivative_Gaussian(x,y,hyp)
	
	g1,f1 = kernels.Derivative_Gaussian_Kron(sample1.grid.W,sample1.grid.dims,sample1.y,hyp,1e-8)
	'''
	
	
	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111, projection='3d')
	
	ax.scatter(sample1.x[:,0],sample1.x[:,1],sample1.y,c='b')
	ax.scatter(sample1.X[:,0],sample1.X[:,1],sample1.mu_s,c='r')
	
	
	
	'''
	demoData = np.load('regression_data.npz')
	x = demoData['x']
	y = demoData['y']
	z = demoData['xstar']
	'''
	
	
	

	
	
	


