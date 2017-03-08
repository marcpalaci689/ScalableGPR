def kron_ind(ind1,ind2,l1):
    ind = np.array([])
    for n in ind1:
        i = n*l1*np.ones((1,ind2.size)) + ind2
        ind = (np.hstack((ind,i)) if ind.size else i)
    return ind
        


if interpolation == 'cubic':
    #self.W = np.zeros((self.N,len(self.X)))
    row_ind = []
    col_ind = np.array([]) 
    weight  = np.array([])          
    index = 0
    
    for n in self.x:
        dim_factor=1
        for d in reversed(xrange(self.D)):
            d1,d2,d3,d4 = 0,0,0,0
            i1,i2,i3,i4 = 0,0,0,0
            dim_factor *= self.gridpoints[d]
            # Initialize searching region. Note that since python rounds down we start at index 1.
            start = 2
            end = self.gridpoints[d]-2
            mid = self.gridpoints[d]/2
            
            # Calculate distance differences at 3 middle points
            diff = n[d]-self.dims[d][mid]
            last_diff = n[d]-self.dims[d][mid - 1]
            next_diff = n[d]-self.dims[d][mid + 1]

            
            if last_diff*diff<0:
                first_diff = n[d] - self.dims[d][mid - 2]  
                d1 = first_diff
                d2 = last_diff
                d3 = diff
                d4 = next_diff
                
                i1 = (mid-2)  
                i2 = (mid-1)  
                i3 = mid 
                i4 = (mid+1) 
                Id = np.array([i1,i2,i3,i4])
                Dd = np.array(CubicInterpolation(d1,d2,d3,d4))
                continue
            elif next_diff*diff<0:
                fourth_diff = n[d] - self.dims[d][mid +2]
                d1 = last_diff
                d2 = diff
                d3 = next_diff
                d4 = fourth_diff
                
                i1 = (mid-1)  
                i2 = mid  
                i3 = (mid+1) 
                i4 = (mid+2) 
                Id = np.array([i1,i2,i3,i4])
                Dd = np.array(CubicInterpolation(d1,d2,d3,d4))
                continue                                        
            elif abs(diff) < 10**-10:
                Id = np.array([mid])
                Dd = np.array([1])  
                continue
            elif abs(last_diff) < 10**-10:    
                Id = np.array([mid-1])
                Dd = np.array([1]) 
                continue                
            elif abs(next_diff) < 10**-10: 
                Id = np.array([mid+1])
                Dd = np.array([1])
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
                    d1 = first_diff
                    d2 = last_diff
                    d3 = diff
                    d4 = next_diff
                    
                    i1 = (mid-2)  
                    i2 = (mid-1)  
                    i3 = mid 
                    i4 = (mid+1) 
                    Id = np.array([i1,i2,i3,i4])
                    Dd = np.array(CubicInterpolation(d1,d2,d3,d4))
                    break
                elif next_diff*diff<0:
                    fourth_diff = n[d] - self.dims[d][mid +2]
                    d1 = last_diff
                    d2 = diff
                    d3 = next_diff
                    d4 = fourth_diff
                    
                    i1 = (mid-1)  
                    i2 = mid  
                    i3 = (mid+1) 
                    i4 = (mid+2) 
                    Id = np.array([i1,i2,i3,i4])
                    Dd = np.array(CubicInterpolation(d1,d2,d3,d4))
                    break                        
                elif abs(diff) < 10**-10:
                    Id = np.array([mid])
                    Dd = np.array([1]) 
                    break
                elif abs(last_diff) < 10**-10:
                    Id = np.array([mid-1])
                    Dd = np.array([1])                                             
                    break
                elif abs(next_diff) < 10**-10:
                    Id = np.array([mid+1])
                    Dd = np.array([1]) 
                    break

            if d == self.D-1:
                col = Id
                row = Dd
            else:
                col = kron_ind(Id,col,dim_factor) 
                row = np.kron(Dd,row_weight)
                
        weight = (np.hstack((weight,row)) if weight.size else row)
        col_ind = (np.hstack((col_ind,col)) if col_ind.size else col)
        row_ind = [index]*len(row)
        index+=1   
    self.W = ss.csr_matrix((weight,(row_ind,col_ind)),shape=(self.N,self.M))
