import numpy as np
from kernels import Gaussian
import math
import scipy.sparse as ss
import time
import solver
from operator import mul
import matplotlib.pyplot as plt

def getMinMax(array):
    ''' 
    Get minimum and maximum value of array simultaneously
    '''
    
    min = array[0]
    max = array[0]
    for i in array[1:]:
        if i>max:
            max = i
        if i<min:
            min = i
    return min,max

def CubicInterpolation(d1,d2,d3,d4):
    ''' 
    Given the distances to the bounding inducing points, get cubic interpolation weights
        
        Inputs:
            d1 --> Distance to furthest lower bound
            d2 --> Distance to closest lower bound
            d3 --> Distance to closest upper bound
            d4 --> Distance to furthest upper bound
    
        Outputs:
            [W1,W2,W3,W4] --> List of weights
    '''
    x1,x2,x3,x4 = 0,d1-d2,d1+d3,d1+d4
    W1 = (d2*-d3*-d4)/(-x4*-x3*-x2)
    W2 = (d1*-d3*-d4)/(x2*(x2-x3)*(x2-x4))
    W3 = (d1*d2*-d4)/(x3*(x3-x2)*(x3-x4))
    W4 = (d1*d2*-d3)/(x4*(x4-x2)*(x4-x3))
    return [W1,W2,W3,W4]
    
def combinations(dimensions):
    '''
    
    This function takes arrays of each dimensions inducing points and then creates 
    a grid (by returning all combinations of coordinates).
    The algorithms starts from the last dimension and works its way up by incrementing
    the indeces.
    
    Inputs:
        dimensions --> List of arrays that is the discretized dimensions
    
    Outputs:
        comb --> matrix consisting of all combinations of the inputs     
    '''
    
    dim = len(dimensions)   # get the number of dimensions
    if dim == 1:
        return dimensions[0].reshape(-1,1)
    ind = np.array([0]*dim) # set indices to zero 
    pointer = dim-1         # set the pointer to the last dimension 
    l = np.array([len(dimensions[i]) for i in xrange(len(dimensions))]) # get the lengths of each dimension
    comb=np.array([])       # initialize combination matrix
    
    while ind[0]<l[0]:  # loop until all values of the first dimension have been used
               
        if ind[-1]<l[-1]: # make sure that the last dimension pointer is not out of range
            x=[]  
            for i in xrange(dim): # get the values corresponding to the indeces
                x.append(dimensions[i][ind[i]]) 
            comb = np.vstack((comb,np.array(x))) if len(comb)>0 else np.array(x) #concatenate to combination matrix          
        
        # increment the last index and subtract index array with length array to see if an index has to be increased
        ind[-1]+=1
        diff = l-ind
        
        # Search for the first zero in the diff array and increase the previous index while reseting the rest
        for i in xrange(dim):
            if diff[i]== 0:
                for j in reversed(xrange(0,i)):
                    if ind[j]+1 < l[j]:
                        ind[j]+=1
                        break 
                    elif j==0:
                        ind[j]+=1 
                ind[j+1:]=[0]*(dim-j-1)
                break        
    return comb    

def MVM(W,K,y):
    ''' 
    This function performs a Matrix-Matrix-Matrix-Vector multiplication
    K_SKI*y = WKW'y
    efficiently in O(N+m^2) time and memory
    
    Inputs:
        W --> Interpolation weights matrix in compressed row format
        K --> mxm grid kernel matrix
        y --> Nx1 target value vector
    
    Outputs:
        WKW'y --> for use in Conjugate Gradient method
    '''
        
    return W.dot(np.dot(K,W.transpose().dot(y)))

def kron_MVM(W,K,y):
    ''' 
    This function performs a Matrix-Matrix-Matrix-Vector multiplication
    K_SKI*y = WKW'y
    efficiently in O(N+m^2) time and memory
    
    Inputs:
        W --> Interpolation weights matrix in compressed row format
        K --> mxm grid kernel matrix
        y --> Nx1 target value vector
    
    Outputs:
        WKW'y --> for use in Conjugate Gradient method
    '''
        
    return W.dot(MV_kronprod(K,W.transpose().dot(y)))    


def MV_kronprod(krons,b):
    '''
    This function will perform a vector matrix product between a packed kronecker matrix
    and a column vector
    
    Inputs:
        krons --> list of tensor matrices
        b     --> Column vector
    
    Outputs:
        x  --> column vector resulting from product between the unpacked kronecker product and
               vector b 
    '''
    x = b
    N = len(b)
    D = len(krons)
    
    for d in reversed(xrange(D)):
        ld = len(krons[d])
        X = x.reshape((ld,N/ld),order='f') 
        Z = np.dot(krons[d],X).T
        x = Z.reshape((-1,1),order='f')
    return x
    

class tensor_grid:
    
    def __init__(self,X,gridpoints):
        
        min = []
        max = []
        self.N = X.shape[0]
        self.M = reduce(mul,gridpoints)
        self.D = X.shape[1]

        # Find min and max values for each dimension
        for d in xrange(self.D):
            minimum,maximum = getMinMax(X[:,d])
            min.append(minimum)
            max.append(maximum)
            
        # set parameters for creating the grid    
        self.params = {'min' : min,
                       'max' : max,
                       'points' : gridpoints}
        #store datapoints
        self.x = X
        self.gridpoints = gridpoints 
    
        
    def generate(self,parameters):
        
        '''
        generate a cartesian grid
        '''
        
        dimensions = len(self.params['min'])
        self.dims = []
        for i in xrange(dimensions):
            delta = (self.params['max'][i] - self.params['min'][i])/(self.params['points'][i]-3)
            self.dims.append(np.hstack((self.params['min'][i]-delta,np.linspace(self.params['min'][i] \
            ,self.params['max'][i],num = self.params['points'][i]-2),self.params['max'][i]+delta)))
        # Get all points and calculate the Gram matrix
        #self.X = combinations(self.dims)
        #self.K = Gaussian(self.X,self.X,1,10)
        
        self.Kd = []
        #self.Kd_inv = []
        for i in self.dims:
            i = i.reshape(-1,1)
            self.Kd.append(Gaussian(i,i,(parameters['sigma'])**(1.0/self.D),parameters['l']))
            #self.Kd_inv.append(np.linalg.inv(self.Kd[-1]))
        
        #self.K = np.kron(self.Kd[1],self.Kd[2])
        #self.K = np.kron(self.Kd[0],self.K)    
            
            
    def SKI(self,interpolation = 'linear'):
        '''
        This function performs a structured kernel interpolation between training points and 
        inducing points that are on a tensor product grid. First it bounds each training point 
        in each dimension by performing a binary search. Then the interpolation weights are 
        determined by the distance between the training points and their bounding inducing points.
        
        Inputs:
           interpolation --> string (either 'linear' or 'cubic') for the type of interpolation
           
        Outputs:
            W --> interpolation weights matrix as a compressed row matrix
            
        STILL TO DO:
        1) incorporate cubic interpolation
        2) reduce memory usage -> DONE
        3) find out how to represent W in compact form -> DONE    
        '''
        
        
        if interpolation == 'linear':
            #self.W = np.zeros((self.N,len(self.X)))
            row_ind = []
            col_ind = [] 
            weight  = []          
            index = 0
            for n in self.x:
                #initialize distances and indeces
                d1,d2 = 0,0
                i1,i2 = 0,0
                # index factor to account for which dimension we are in
                dimensional_factor = 1
                     
                for d in reversed(xrange(self.D)):
                    # Initialize searching region. Note that since python rounds down we start at index 1.
                    start = 1
                    end = self.gridpoints[d]-1
                    mid = self.gridpoints[d]/2
                    
                    # Calculate distance differences at 3 middle points
                    diff = n[d]-self.dims[d][mid]
                    last_diff = n[d]-self.dims[d][mid - 1]
                    next_diff = n[d]-self.dims[d][mid + 1]
                    
                    # Calculate dimensional factor
                    if d != self.D-1:
                        dimensional_factor*=self.gridpoints[d+1]
                    
                    if last_diff*diff<0:
                        d1+=last_diff**2
                        d2+=diff**2
                        i1 += (mid-1)*dimensional_factor 
                        i2 += mid*dimensional_factor 
                        continue
                    elif next_diff*diff<0:
                        d1+=diff**2
                        d2+=next_diff**2
                        i1 += (mid)*dimensional_factor 
                        i2 += (mid+1)*dimensional_factor 
                        continue                                        
                    elif abs(diff) < 10**-10:
                        i1 += mid*dimensional_factor  
                        i2 += mid*dimensional_factor  
                        continue
                    elif abs(last_diff) < 10**-10:
                        i1 += (mid-1)*dimensional_factor  
                        i2 += (mid-1)*dimensional_factor  
                        continue                
                    elif abs(next_diff) < 10**-10:
                        i1 += (mid+1)*dimensional_factor  
                        i2 += (mid+1)*dimensional_factor  
                        continue 
                                        
                    while last_diff*diff>0:
                        
                        # Update searching section
                        if diff<0:
                            end = mid
                            mid = start + (mid-start)/2    
                        else:
                            start = mid
                            mid = mid + (end-mid)/2
                        
                        # calculate new distance differences at middle points
                        diff = n[d] - self.dims[d][mid]
                        last_diff = n[d]-self.dims[d][mid - 1]
                        next_diff = n[d]-self.dims[d][mid + 1]
                        
                        # Break if one of the following conditions are met
                        if last_diff*diff<0:
                            d1+=last_diff**2
                            d2+=diff**2
                            i1 += (mid-1)*dimensional_factor 
                            i2 += mid*dimensional_factor 
                            break
                        elif next_diff*diff<0:
                            d1+=diff**2
                            d2+=next_diff**2
                            i1 += (mid)*dimensional_factor 
                            i2 += (mid+1)*dimensional_factor 
                            break                        
                        elif abs(diff) < 10**-10:
                            i1 += mid*dimensional_factor  
                            i2 += mid*dimensional_factor 
                            break
                        elif abs(last_diff) < 10**-10:
                            i1 += (mid-1)*dimensional_factor  
                            i2 += (mid-1)*dimensional_factor                                                
                            break
                        elif abs(next_diff) < 10**-10:
                            i1 += (mid+1)*dimensional_factor  
                            i2 += (mid+1)*dimensional_factor                                                
                            break
                     

                # Record distances         
                d1 = math.sqrt(d1)
                d2 = math.sqrt(d2)           
      
          
                if d1==0 or d2 == 0: 
                    weight.append(1)
                    row_ind.append(index)
                    col_ind.append(i1)   
                else:  
                    weight.append((d1**-1)/(d1**-1+d2**-1))
                    row_ind.append(index)
                    col_ind.append(i1)  
                    
                    weight.append(1 - (d1**-1)/(d1**-1+d2**-1))
                    row_ind.append(index)
                    col_ind.append(i2) 
 
                index+=1
                
            self.W = ss.csr_matrix((weight,(row_ind,col_ind)),shape=(self.N,self.M))

        if interpolation == 'cubic':
            #self.W = np.zeros((self.N,len(self.X)))
            row_ind = []
            col_ind = [] 
            weight  = []          
            index = 0
            for n in self.x:
                #initialize distances and indeces
                d1,d2,d3,d4 = 0,0,0,0
                i1,i2,i3,i4 = 0,0,0,0
                
                # index factor to account for which dimension we are in
                dimensional_factor = 1
                     
                for d in reversed(xrange(self.D)):
                    # Initialize searching region. Note that since python rounds down we start at index 1.
                    start = 2
                    end = self.gridpoints[d]-2
                    mid = self.gridpoints[d]/2
                    
                    # Calculate distance differences at 3 middle points
                    diff = n[d]-self.dims[d][mid]
                    last_diff = n[d]-self.dims[d][mid - 1]
                    next_diff = n[d]-self.dims[d][mid + 1]
                    
                    # Calculate dimensional factor
                    if d != self.D-1:
                        dimensional_factor*=self.gridpoints[d + 1]
                    
                    if last_diff*diff<0:
                        first_diff = n[d] - self.dims[d][mid - 2]  
                        d1 += first_diff**2
                        d2 += last_diff**2
                        d3 += diff**2
                        d4 += next_diff**2
                        
                        i1 += (mid-2)*dimensional_factor 
                        i2 += (mid-1)*dimensional_factor 
                        i3 += mid*dimensional_factor
                        i4 += (mid+1)*dimensional_factor
                        continue
                    elif next_diff*diff<0:
                        fourth_diff = n[d] - self.dims[d][mid +2]
                        d1 += last_diff**2
                        d2 += diff**2
                        d3 += next_diff**2
                        d4 += fourth_diff**2
                        
                        i1 += (mid-1)*dimensional_factor 
                        i2 += mid*dimensional_factor 
                        i3 += (mid+1)*dimensional_factor
                        i4 += (mid+2)*dimensional_factor
                        continue                                        
                    elif abs(diff) < 10**-10:
                        i1 += mid*dimensional_factor  
                        i2 += mid*dimensional_factor
                        i3 += mid*dimensional_factor  
                        i4 += mid*dimensional_factor   
                        continue
                    elif abs(last_diff) < 10**-10:
                        i1 += (mid-1)*dimensional_factor  
                        i2 += (mid-1)*dimensional_factor
                        i3 += (mid-1)*dimensional_factor  
                        i4 += (mid-1)*dimensional_factor      
                        continue                
                    elif abs(next_diff) < 10**-10:
                        i1 += (mid+1)*dimensional_factor  
                        i2 += (mid+1)*dimensional_factor  
                        i3 += (mid+1)*dimensional_factor  
                        i4 += (mid+1)*dimensional_factor 
                        continue 
                                        
                    while last_diff*diff>0:
                        
                        # Update searching section
                        if diff<0:
                            end = mid
                            mid = start + (mid-start)/2    
                        else:
                            start = mid
                            mid = mid + (end-mid)/2
                        
                        # calculate new distance differences at middle points
                        diff = n[d] - self.dims[d][mid]
                        last_diff = n[d]-self.dims[d][mid - 1]
                        next_diff = n[d]-self.dims[d][mid + 1]
                        
                        # Break if one of the following conditions are met
                        if last_diff*diff<0:
                            first_diff = n[d] - self.dims[d][mid - 2]  
                            d1 += first_diff**2
                            d2 += last_diff**2
                            d3 += diff**2
                            d4 += next_diff**2
                            
                            i1 += (mid-2)*dimensional_factor 
                            i2 += (mid-1)*dimensional_factor 
                            i3 += mid*dimensional_factor
                            i4 += (mid+1)*dimensional_factor
                            break
                        elif next_diff*diff<0:
                            fourth_diff = n[d] - self.dims[d][mid +2]
                            d1 += last_diff**2
                            d2 += diff**2
                            d3 += next_diff**2
                            d4 += fourth_diff**2
                            
                            i1 += (mid-1)*dimensional_factor 
                            i2 += mid*dimensional_factor 
                            i3 += (mid+1)*dimensional_factor
                            i4 += (mid+2)*dimensional_factor
                            break                        
                        elif abs(diff) < 10**-10:
                            i1 += mid*dimensional_factor  
                            i2 += mid*dimensional_factor
                            i3 += mid*dimensional_factor  
                            i4 += mid*dimensional_factor 
                            break
                        elif abs(last_diff) < 10**-10:
                            i1 += (mid-1)*dimensional_factor  
                            i2 += (mid-1)*dimensional_factor
                            i3 += (mid-1)*dimensional_factor  
                            i4 += (mid-1)*dimensional_factor                                                 
                            break
                        elif abs(next_diff) < 10**-10:
                            i1 += (mid+1)*dimensional_factor  
                            i2 += (mid+1)*dimensional_factor  
                            i3 += (mid+1)*dimensional_factor  
                            i4 += (mid+1)*dimensional_factor                                                
                            break
                     

                # Record distances         
                d1,d2,d3,d4 = math.sqrt(d1),math.sqrt(d2),math.sqrt(d3),math.sqrt(d4)
                                    
                if d1==0 or d2 == 0 or d3==0 or d4==0: 
                    weight.append(1)
                    row_ind.append(index)
                    col_ind.append(i1)   
                else:  
                    weight += CubicInterpolation(d1,d2,d3,d4)
                    row_ind += [index]*4
                    col_ind += [i1,i2,i3,i4]
                
                index+=1
            self.weight = weight     
            self.W = ss.csr_matrix((weight,(row_ind,col_ind)),shape=(self.N,self.M))
         

                                                   
if __name__ == '__main__':
 
    parameters = { 's':0.0001,
                    'sigma' : 1,
                    'l':25}
    x = np.sort(np.random.normal(scale=25,size=(1,1000))).T
    grid = tensor_grid(x,[40])   
    grid.generate(parameters)
    start = time.time()
    grid.SKI(interpolation='cubic')
    end = time.time()
    print('done in %2.16f seconds' %(end-start))   
    
   
    grid.y = np.random.normal(scale=2,size=(100,1))
    grid.K = grid.Kd[0]
    K_SKI = grid.W.dot((grid.W.dot(grid.K)).T).T     
    K = Gaussian(x,x,1,25)
    
    '''
    plt.figure(1)
    plt.contourf(K,100)
    plt.colorbar()
    
    plt.figure(2)
    plt.contourf(K_SKI,100)
    plt.colorbar()
    '''
    plt.figure(1)
    plt.contourf(np.abs(K-K_SKI),100)
    plt.colorbar()
   
    
    