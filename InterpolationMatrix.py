import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import sample
import warnings

warnings.filterwarnings("ignore")

if sys.version_info.major == 3:
	xrange = range

def EuclideanDistance(a,b):
	return np.sum(a**2,1).reshape(-1,1)+np.sum(b**2,1)-2*np.dot(a,b.T)
	
def Interpolate(X,U):

	#Perform a linear interpolation matrix
	'''
	Inputs: 	X --> Matrix of points to predict = N x d
				U --> Matrix of inducing points = M x d 

	Outputs:    W --> interpolation matrix --> N x M
	 
	TO DO: compress W (very easy to do) !!!!!!!!
	'''
	(N,D) = X.shape
	M = U.shape[0]
	I = np.ones((M,1))
	W = np.zeros((N,M))
	num=0 	
	for P in X:
		#first find closest point
		# get euclidean distances
		dist= EuclideanDistance(np.array([P]),U)[0]
		# find fursthest distance
		largest_val = max(dist)
		# find index of closest point
		closest = np.argmin(dist)
		closest_dist = dist[closest]**0.5
		# if we can interpolate exactly:
		if abs(closest_dist) <=10^-3:
			W[num,closest]=1
			num+=1
			continue
 		# Replace the minimum distance with the value of the furthest distance
		dist[closest] =  largest_val
		store_close = np.argmin(dist)
		store_close_dist = dist[store_close]
		for i in xrange(M//2):
			# Find next closest point
			close = np.argmin(dist)
			# Evaluate dot product and ensure it is less than 0 (to bound the point)
			
			if (np.dot((U[close]-P).T,(U[closest]-P)))<0:
				close_dist = dist[close]**0.5
				break
			else:
				dist[close] = largest_val
			

			### TESTING WITH THE FOLLOWING TWO LINES
			'''
			close_dist = dist[close]**0.5
			break
			'''
			close = store_close**0.5
			close_dist = store_close_dist
		### We can easily compress the w matrix here, but for now keep it sparse
		
		### Sanity check for numpy glitches
		if closest_dist==0.0 or np.isnan(closest_dist):
			W[num,closest]=1
			num+=1
			continue

		w = (closest_dist**-1)/(closest_dist**-1 + close_dist**-1)
		#w = math.exp(-closest_dist**2)/(math.exp(-closest_dist**2)+math.exp(-close_dist**2))
		W[num,closest] = w
		W[num,close] = 1.0 - w
		num+=1
	return W	
			

def InducingPoints(x,m):
	N = x.shape[0]
	U = x[sample.RandomSample(0,N-1,size=m)]
	return U







