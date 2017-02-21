import numpy as np
import scipy.sparse as ss
import time
import warnings
import math


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

def Derivative_Gaussian(x1,x2,param1,param2, param3):
	sigma = math.exp(-param1)
	l = math.exp(-param2) 
	s = math.exp(-param3)
	K =  (sigma**2)*np.exp(-(np.sum(x1**2,1).reshape(-1,1)+np.sum(x2**2,1)-2*np.dot(x1,x2.T))/(2.0*l**2))
	dK_dsigma = -2.0*K
	dK_dl     =  -(1.0/l**2)*np.multiply(K,(np.sum(x1**2,1).reshape(-1,1)+np.sum(x2**2,1)-2*np.dot(x1,x2.T)))
	dK_ds     = -(2*s**2)*np.eye(len(x1))
	K = K + (s**2)*np.eye(len(x1))
	return K, dK_dsigma, dK_dl, dK_ds


	
