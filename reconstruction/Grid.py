import numpy as np
import math
import scipy.sparse as ss
import time
from operator import mul
import matplotlib.pyplot as plt


def kron_ind(ind1,ind2,l1):
    ind = np.array([])
    for n in ind1:
        i = n*l1*np.ones((1,ind2.size)) + ind2
        ind = (np.hstack((ind,i)) if ind.size else i)
    return ind

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
    


class Grid:
    
    def __init__(self,X,gridpoints):
        
        '''
        generate a cartesian grid
        '''        
        
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
        self.gridpoints = gridpoints 
        
        
        dimensions = len(self.params['min'])
        self.x = []
        for i in xrange(dimensions):
            delta = (self.params['max'][i] - self.params['min'][i])/(self.params['points'][i]-3)
            self.x.append(np.hstack((self.params['min'][i]-delta,np.linspace(self.params['min'][i] \
            ,self.params['max'][i],num = self.params['points'][i]-2),self.params['max'][i]+delta)))
        
        return
 
            
    def Interpolate(self,x,scheme='linear'):
        '''
        This function performs a structured kernel interpolation between training points and 
        inducing points that are on a tensor product grid. First it bounds each training point 
        in each dimension by performing a binary search. Then the interpolation weights are 
        determined by the distance between the training points and their bounding inducing points.
        
        Inputs:
           scheme --> string (either 'linear', 'ModifiedSheppard' or 'cubic') for the type of interpolation
           
        Outputs:
            W --> interpolation weights matrix as a compressed row matrix
            
        STILL TO DO:
        1) incorporate cubic interpolation
        2) reduce memory usage -> DONE
        3) find out how to represent W in compact form -> DONE    
        '''
        
        
        if scheme == 'linear':
            #self.W = np.zeros((self.N,len(self.X)))
            row_ind = []
            col_ind = [] 
            weight  = []          
            index = 0
            for n in x:
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
                    diff = n[d]-self.x[d][mid]
                    last_diff = n[d]-self.x[d][mid - 1]
                    next_diff = n[d]-self.x[d][mid + 1]
                    
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
                        diff = n[d] - self.x[d][mid]
                        last_diff = n[d]-self.x[d][mid - 1]
                        next_diff = n[d]-self.x[d][mid + 1]
                        
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
                
            return ss.csr_matrix((weight,(row_ind,col_ind)),shape=(self.N,self.M))

        if scheme == 'ModifiedSheppard':
            #self.W = np.zeros((self.N,len(self.X)))
            
            row_ind = []
            col_ind = [] 
            weight  = []          
            index = 0
            for n in x:
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
                    diff = n[d]-self.x[d][mid]
                    last_diff = n[d]-self.x[d][mid - 1]
                    next_diff = n[d]-self.x[d][mid + 1]
                    
                    # Calculate dimensional factor
                    if d != self.D-1:
                        dimensional_factor*=self.gridpoints[d + 1]
                    
                    if last_diff*diff<0:
                        first_diff = n[d] - self.x[d][mid - 2]  
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
                      
                        fourth_diff = n[d] - self.x[d][mid +2]
                        d1 += last_diff**2
                        d2 += diff**2
                        d3 += next_diff**2
                        d4 += fourth_diff**2
                        
                        i1 += (mid-1)*dimensional_factor 
                        i2 += mid*dimensional_factor 
                        i3 += (mid+1)*dimensional_factor
                        i4 += (mid+2)*dimensional_factor
                        continue                                        
                    elif abs(diff) < 10**-15:
                        i1 += mid*dimensional_factor  
                        i2 += mid*dimensional_factor
                        i3 += mid*dimensional_factor  
                        i4 += mid*dimensional_factor   
                        continue
                    elif abs(last_diff) < 10**-15:
                        i1 += (mid-1)*dimensional_factor  
                        i2 += (mid-1)*dimensional_factor
                        i3 += (mid-1)*dimensional_factor  
                        i4 += (mid-1)*dimensional_factor      
                        continue                
                    elif abs(next_diff) < 10**-15:
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
                        diff = n[d] - self.x[d][mid]
                        last_diff = n[d]-self.x[d][mid - 1]
                        next_diff = n[d]-self.x[d][mid + 1]
                        
                        # Break if one of the following conditions are met
                        if last_diff*diff<0:
                            first_diff = n[d] - self.x[d][mid - 2]  
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
                            fourth_diff = n[d] - self.x[d][mid +2]
                            d1 += last_diff**2
                            d2 += diff**2
                            d3 += next_diff**2
                            d4 += fourth_diff**2
                            
                            i1 += (mid-1)*dimensional_factor 
                            i2 += mid*dimensional_factor 
                            i3 += (mid+1)*dimensional_factor
                            i4 += (mid+2)*dimensional_factor
                            break                        
                        elif abs(diff) < 10**-15:
                            i1 += mid*dimensional_factor  
                            i2 += mid*dimensional_factor
                            i3 += mid*dimensional_factor  
                            i4 += mid*dimensional_factor 
                            break
                        elif abs(last_diff) < 10**-15:
                            i1 += (mid-1)*dimensional_factor  
                            i2 += (mid-1)*dimensional_factor
                            i3 += (mid-1)*dimensional_factor  
                            i4 += (mid-1)*dimensional_factor                                                 
                            break
                        elif abs(next_diff) < 10**-15:
                            i1 += (mid+1)*dimensional_factor  
                            i2 += (mid+1)*dimensional_factor  
                            i3 += (mid+1)*dimensional_factor  
                            i4 += (mid+1)*dimensional_factor                                                
                            break
                     
                                    
                if d1==0 or d2 == 0 or d3==0 or d4==0: 
                    weight.append(1)
                    row_ind.append(index)
                    col_ind.append(i1)   
                else:  
                    weight += CubicInterpolation(d1**0.5,d2**0.5,d3**0.5,d4**0.5)
                    row_ind += [index]*4
                    col_ind += [i1,i2,i3,i4]
                
                index+=1
  
            return ss.csr_matrix((weight,(row_ind,col_ind)),shape=(self.N,self.M))
        
        if scheme == 'cubic':
       
            row_ind = []
            col_ind = np.array([]) 
            weight  = np.array([])          
            index = 0
    
            for n in x:
                dim_factor=1
                for d in reversed(xrange(self.D)):
                    if d!=self.D-1:    
                        dim_factor *= self.gridpoints[d]
                    # Initialize searching region. Note that since python rounds down we start at index 1.
                    start = 2
                    end = self.gridpoints[d]-2
                    mid = self.gridpoints[d]/2
                    
                    # Calculate distance differences at 3 middle points
                    diff = n[d]-self.x[d][mid]
                    last_diff = n[d]-self.x[d][mid - 1]
                    next_diff = n[d]-self.x[d][mid + 1]
        
                    
                    if last_diff*diff<0:
                        first_diff = n[d] - self.x[d][mid - 2]  
                        d1 = first_diff**2
                        d2 = last_diff**2
                        d3 = diff**2
                        d4 = next_diff**2
                        
                        i1 = (mid-2)  
                        i2 = (mid-1)  
                        i3 = mid 
                        i4 = (mid+1) 
                        Id = np.array([i1,i2,i3,i4])
                        Dd = np.array(CubicInterpolation(d1**0.5,d2**0.5,d3**0.5,d4**0.5))
                        
                    
                    elif next_diff*diff<0:
                    
                        fourth_diff = n[d] - self.x[d][mid +2]
                        d1 = last_diff**2
                        d2 = diff**2
                        d3 = next_diff**2
                        d4 = fourth_diff**2
                        
                        i1 = (mid-1)  
                        i2 = mid  
                        i3 = (mid+1) 
                        i4 = (mid+2) 
                        Id = np.array([i1,i2,i3,i4])
                        Dd = np.array(CubicInterpolation(d1**0.5,d2**0.5,d3**0.5,d4**0.5))
                                                                
                    elif abs(diff) < 10**-15:
                        Id = np.array([mid])
                        Dd = np.array([1])  
                        
                    elif abs(last_diff) < 10**-15:    
                        Id = np.array([mid-1])
                        Dd = np.array([1]) 
                                       
                    elif abs(next_diff) < 10**-15: 
                        Id = np.array([mid+1])
                        Dd = np.array([1])
                        
                    
                    else:           
                        while last_diff*diff>0:
                            
                            # Update searching section
                            if diff<0:
                                end = mid
                                mid = start + (mid-start)/2    
                            else:
                                start = mid
                                mid = mid + (end-mid)/2
                            
                            # calculate new distance differences at middle points
                            diff = n[d] - self.x[d][mid]
                            last_diff = n[d]-self.x[d][mid - 1]
                            next_diff = n[d]-self.x[d][mid + 1]
                            
                            # Break if one of the following conditions are met
                            if last_diff*diff<0:
                                first_diff = n[d] - self.x[d][mid - 2]  
                                d1 = first_diff**2
                                d2 = last_diff**2
                                d3 = diff**2
                                d4 = next_diff**2
                                
                                i1 = (mid-2)  
                                i2 = (mid-1)  
                                i3 = mid 
                                i4 = (mid+1) 
                                Id = np.array([i1,i2,i3,i4])
                                Dd = np.array(CubicInterpolation(d1**0.5,d2**0.5,d3**0.5,d4**0.5))
                                break
                            elif next_diff*diff<0:
                                fourth_diff = n[d] - self.x[d][mid +2]
                                d1 = last_diff**2
                                d2 = diff**2
                                d3 = next_diff**2
                                d4 = fourth_diff**2
                                
                                i1 = (mid-1)  
                                i2 = mid  
                                i3 = (mid+1) 
                                i4 = (mid+2) 
                                Id = np.array([i1,i2,i3,i4])
                                Dd = np.array(CubicInterpolation(d1**0.5,d2**0.5,d3**0.5,d4**0.5))
                                break                        
                            elif abs(diff) < 10**-15:
                                Id = np.array([mid])
                                Dd = np.array([1]) 
                                break
                            elif abs(last_diff) < 10**-15:
                                Id = np.array([mid-1])
                                Dd = np.array([1])                                             
                                break
                            elif abs(next_diff) < 10**-15:
                                Id = np.array([mid+1])
                                Dd = np.array([1]) 
                                break
        
                    if d == self.D-1:
                        col = Id
                        row = Dd
                    else:
                        col = kron_ind(Id,col,dim_factor) 
                        row = np.kron(Dd,row)
                        
                weight = (np.hstack((weight,row)) if weight.size else row)
                col_ind = (np.hstack((col_ind,col)) if col_ind.size else col)
                row_ind += [index]*len(row)
                index+=1  
            col_ind = col_ind.reshape(-1,) 
 
            return ss.csr_matrix((weight,(row_ind,col_ind)),shape=(self.N,self.M))
                 

                                                   
if __name__ == '__main__':
    '''
    parameters = { 's':11,
                    'sigma' : 0,
                    'l':-3}
    x1 = np.sort(np.random.normal(scale=25,size=(1,1000))).T
    x2 = np.sort(np.random.normal(scale=25,size=(1,1000))).T
    #x3 = np.sort(np.random.normal(scale=25,size=(1,1000))).T
    x = np.hstack((x1,x2))
    grid = tensor_grid(x,[40,40])   
    grid.generate(parameters)
    start = time.time()
    grid.SKI(scheme='cubic')
    end = time.time()
    print('done in %2.16f seconds' %(end-start))   
    
    noise = math.exp(-parameters['s'])
   
    grid.y = np.random.normal(scale=2,size=(100,1))
    
    K_SKI = unpack.KSKI_Unpack(grid.W,grid.Kd) + (noise**2)*np.eye(1000)  
    K = Gaussian(x,x,0,-3,10)
    
    
    
    grid1 = tensor_grid(x,[40,40])   
    grid1.generate(parameters)
    start = time.time()
    grid1.SKI(scheme='ModifiedSheppard')
    end = time.time()
    print('done in %2.16f seconds' %(end-start))   
    
    noise = math.exp(-parameters['s'])
    
    
    grid1.y = np.random.normal(scale=2,size=(100,1))
    K_SKI1 = unpack.KSKI_Unpack(grid1.W,grid1.Kd)    + (noise**2)*np.eye(1000)  


    
    plt.figure(1)
    plt.clf()
    plt.contourf(K,100)
    plt.colorbar()
    
    plt.figure(2)
    plt.clf()
    plt.contourf(K_SKI,100)
    plt.colorbar()
    
    plt.figure(3)
    plt.clf()
    plt.contourf(np.abs(K-K_SKI),100)
    plt.gca().invert_yaxis()
    plt.colorbar()
    
    plt.figure(4)
    plt.clf()
    plt.contourf(np.abs(K-K_SKI1),100)
    plt.gca().invert_yaxis()
    plt.colorbar()
    '''
