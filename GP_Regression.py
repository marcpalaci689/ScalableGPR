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
			while iter<1:
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
			
			params = 1.0*np.array([[self.sigma],[self.l],[self.s]])
			min = kernels.Gaussian_Kron(self.grid.W,self.grid.dims,self.y,params)
			if min==-np.inf: min=np.inf
			best_params = params
			print(min)
			
			params,ML,i = solver.minimize_kron(params,self.grid.W,self.grid.dims,self.y,verbose=True)
			print(ML[-1])
			if ML[-1] != -np.inf and ML[-1] < min :
				print(len(ML))
				min = ML[-1]
				best_params = params		
			
			iter= 1
			while iter<1:
				params = 0.2*np.random.randint(-20,20,size=(3,1))
				params,ML,i = solver.minimize_kron(params,self.grid.W,self.grid.dims,self.y,verbose=True)
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
		self.grid 	 = kron.tensor_grid(self.x,[100,100])
		
		# Generate grid
		self.grid.generate(self.parameters)	
		
		#Perform Interpolation
		self.grid.SKI(interpolation='ModifiedSheppard')
		
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
	N = 2500
	data = np.load('test_data.npz')
	x1 = np.random.normal(scale=4,size=(N,1))
	x2 = np.random.normal(scale=4,size=(N,1))
	x1s = np.linspace(-1,51,num=500).reshape(500,1)
	x2s = np.linspace(-1,51,num=500).reshape(500,1)
	x = np.hstack((x1,x2))
	xs=np.hstack((x1s,x2s))
	y= 10*np.sin(0.1*(x1+x2)).reshape(N,1)
	parameters={ 'dataset':'Modified',
				 'n':N,
		         's': 5.0,
		         'sigma': -1.0,
		         'l': -2.0 ,
		         'kernel': 'SE',
		         'x' : x,
		         'y' : y,
		         'X' : xs,
		         }
	

	hyp = np.array([[parameters['sigma']],[parameters['l']],[parameters['s']]])
	
	sample = GPRegression(parameters)
	sample.Inducing_Points()
	
	sample.KISS_GP() 
	
	
	g1,f1 = kernels.Derivative_Gaussian_Kron(sample.grid.W,sample.grid.dims,sample.y,hyp,epsilon=1e-4)
	g,f = kernels.Derivative_Gaussian(sample.x,sample.y,hyp)
	
	noise = math.exp(-parameters['s'])
	#K = np.kron(sample.grid.Kd[0],sample.grid.Kd[1])
	#K_SKI = sample.grid.W.dot((sample.grid.W.dot(K)).T).T + (noise**2)*np.eye(2500)
	#Ky = kernels.Gaussian(x,x,parameters['sigma'],parameters['l'],parameters['s'])
	#print(np.linalg.norm(Ky-K_SKI)/np.linalg.norm(Ky))
	
	#e  = np.real(np.linalg.eigh(K_SKI)[0]) 
	#e1 = np.real(np.linalg.eigh(sample.grid.Kd[0])[0])
	#e2 = np.real(np.linalg.eigh(sample.grid.Kd[1])[0])
	#ea = np.kron(e1,e2)