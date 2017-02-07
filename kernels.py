import numpy as np
import scipy.sparse as ss
import time
import warnings


warnings.filterwarnings("ignore",category=DeprecationWarning)
### Gaussian kernel function
def Gaussian(x1,x2,sigma,l):
	return (sigma**2)*np.exp(-(np.sum(x1**2,1).reshape(-1,1)+np.sum(x2**2,1)-2*np.dot(x1,x2.T))/(2*l**2))

