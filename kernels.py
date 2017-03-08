import numpy as np
import scipy.sparse as ss
import time
import warnings
import math
import solver

warnings.filterwarnings("ignore",category=DeprecationWarning)
def NoNoise_Gaussian(x1,x2,param1,param2):
	sigma = math.exp(-param1)
	l = math.exp(-param2) 
	return (sigma**2.0)*np.exp(-(np.sum(x1**2,1).reshape(-1,1)+np.sum(x2**2,1)-2*np.dot(x1,x2.T))/(2.0*l**2))

def Gaussian(x1,x2,param1,param2,param3):
	sigma = math.exp(-param1)
	l = math.exp(-param2) 
	s = math.exp(-param3)
	return (sigma**2.0)*np.exp(-(np.sum(x1**2,1).reshape(-1,1)+np.sum(x2**2,1)-2*np.dot(x1,x2.T))/(2.0*l**2)) + (s**2)*np.eye(len(x1))

def Derivative_Gaussian(x,y,hyp):
	
	N=len(x)
	sigma = math.exp(-hyp[0])
	l = math.exp(-hyp[1]) 
	s = math.exp(-hyp[2])

	K =  (sigma**2)*np.exp(-(np.sum(x**2,1).reshape(-1,1)+np.sum(x**2,1)-2*np.dot(x,x.T))/(2.0*l**2))
	dK_dsigma = -2.0*K
	dK_dl     =  -(1.0/l**2)*np.multiply(K,(np.sum(x**2,1).reshape(-1,1)+np.sum(x**2,1)-2*np.dot(x,x.T)))
	dK_ds     = -(2*s**2)*np.eye(len(x))
	K = K + (s**2)*np.eye(len(x))
	
	
	L = np.linalg.cholesky(K)
	inv_K = np.dot(np.linalg.inv(L).T,np.linalg.inv(L))
	alpha = np.linalg.solve(L.T,np.linalg.solve(L,y))
	
	
	d_sigma = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_K),dK_dsigma))
	d_l = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_K),dK_dl))
	d_s = 0.5*np.trace(np.dot((np.dot(alpha,alpha.T)-inv_K),dK_ds))
	grad = -np.array([[d_sigma],[d_l],[d_s]])
	
	
	complexity = sum(2*np.log(np.diag(L)))
	'''
	print('YTalpha = %.8f ' %(np.dot(y.T,alpha)[0][0]))
	print('complexity = %.8f' %(complexity))
	'''
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
	alpha = solver.Linear_CG(W,K,y,s,tolerance=1e-8)
	alpha = alpha[0]
	'''
	print('YTalpha = %.8f ' %(np.dot(y.T,alpha)[0][0]))
	print('complexity = %.8f' %(complexity))
	'''
	# Get negative log likelihood (objective function to be minimized)
	return 0.5*(np.dot(y.T,alpha)[0][0] + complexity + N*np.log(2*math.pi))	

def Derivative_Gaussian_Kron(W,x,y,hyp,epsilon=1e-3):
	
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
			
		

	
	
