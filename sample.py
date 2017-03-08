import random
import sys
import numpy as np

if sys.version_info.major == 3:
	xrange = range

def RandomSample(low,high,size=1):
	sample = []
	choices = list(xrange(low,high+1))
	if size>len(choices):
		raise ValueError("Number of samples exceeds number data points")
	for i in xrange(size):
		ran = random.randint(0,len(choices)-1)
		sample.append(choices[ran])
		choices.pop(ran)
	return np.array(sample)

def SeperateData(x,y,testsize=1):
	maximum = x.shape[0]-1 
	test_x = []
	test_y = []
	for i in xrange(testsize):
		sel = random.randint(0,maximum)
		test_x.append(x[sel])
		test_y.append(y[sel])
		x=np.delete(x,sel,0)
		y=np.delete(y,sel,0)
		maximum-=1
	return x,y,np.array(test_x),np.array(test_y)


def NormalizeData(train,test):
	D = train.shape[1]
	for d in xrange(D):
		normalizer = np.max(train[:,d])-np.min(train[:,d])
		train[:,d] = 1/normalizer*train[:,d]
		test[:,d] = 1/normalizer*test[:,d]
	return train,test

