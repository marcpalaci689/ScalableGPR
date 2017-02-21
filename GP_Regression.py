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

	
	def optimizeHyperparameters(self):
		
		''' for testing purposes '''
		K_y = Gaussian(self.x,self.x,self.sigma,self.l,self.s)
		### Calculate posterior mean and covariance 
		### NOTE: we are assuming mean is zero 
		L            = np.linalg.cholesky(K_y)
		self.alpha   = np.linalg.solve(L.T,np.linalg.solve(L,self.y))
		self.marginal_likelihood = -0.5*np.dot(self.y.T,self.alpha) - \
		0.5*np.log(np.linalg.det(K_y)) -0.5*self.N*np.log(2*math.pi)
		
		print(self.marginal_likelihood)
		''' end of testing purpose '''
		
		iter = 1
		params = np.array([[self.sigma],[self.l],[self.s]])
		try:
			opt = solver.NonLinear_CG(self.x,self.y,params,maxiter=100)
			best = opt[0]
			max = opt[1][0]
		except:
			opt = np.ones((3,1))
			best = params
			max = self.marginal_likelihood

	
		while opt[2] != 0 and iter<=3:	
			params = np.random.normal(loc = 2,scale=2.5,size=(3,1))
			try:
				opt = solver.NonLinear_CG(self.x,self.y,params,maxiter=200)
				if opt[1][0] > max or opt[2] == 0:
					if opt[2] == 0:
						print('solution found')
					max = opt[1][0]
					best = opt[0]
					break
			except:	
				iter+=1
		

		self.sigma = best[0]
		self.l     = best[1]
		self.s     = best[2]	
		self.parameters['sigma'] = self.sigma [0]
		self.parameters['l'] = self.l[0]
		self.parameters['s'] = self.s[0]
		print(max)

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


	def KISS_GP(self):
		
		# Initialize grid
		self.grid 	 = kron.tensor_grid(self.x,[50,50])
		
		# Generate grid
		self.grid.generate(self.parameters)	
		
		#Perform Interpolation
		self.grid.SKI(interpolation='cubic')
		
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
	x1 = np.linspace(0,100,num=100).reshape(100,1)
	x2 = np.linspace(0,100,num=100).reshape(100,1)
	x1s = np.linspace(-10,110,num=200).reshape(200,1)
	x2s = np.linspace(-10,110,num=200).reshape(200,1)
	x = np.hstack((x1,x2))
	xs=np.hstack((x1s,x2s))
	y= 100*np.sin(x1+x2).reshape(100,1)
	parameters={ 'dataset':'Modified',
				 'n':200,
		         's':1,
		         'sigma':1,
		         'l': 1,
		         'kernel': 'SE',
		         'x' : x,
		         'y' : y,
		         'X' : xs,
		         }
	
	sample = GPRegression(parameters)
	sample.optimizeHyperparameters()
	sample.GP_Regression()
	
	sample1 = GPRegression(parameters)
	#sample1.optimizeHyperparameters()
	sample1.KISS_GP() 
	Kski = np.kron(sample1.grid.Kd[0],sample1.grid.Kd[1])
	Kski = sample1.grid.W.dot((sample1.grid.W.dot(Kski)).T).T + math.exp(-2*sample1.s)*np.eye(len(sample1.x))
	Ky = Gaussian(sample1.x,sample1.x,sample1.sigma,sample1.l,sample1.s)
	#sample.GP_Regression()
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	ax.scatter(sample.x[:,0],sample.x[:,1],sample.y,c='b')
	ax.scatter(sample.X[:,0],sample.X[:,1],sample.mu_s,c='r')
	
	
	
	'''
	demoData = np.load('regression_data.npz')
	x = demoData['x']
	y = demoData['y']
	z = demoData['xstar']
	'''
	
	
	

	
	
	


