import numpy as np
import scipy.sparse as ss
import kernels 
import scipy.sparse.linalg as solv 
import time
import math
from numpy import dot, isinf, isnan, any, sqrt, isreal, real, nan, inf

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
    

def kron_MVM(W,K,y,noise):
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
        
    return W.dot(MV_kronprod(K,W.transpose().dot(y))) + (noise**2)*y   


def Linear_CG(W,K,y,noise,tolerance=1e-8,maxiter=10000):
    ''' Conjugate Gradient method to solve for alpha:
    alpha = inverse(WkW')*y
    
    Inputs:
        W --> Nxm weight matrix in sparse matrix format
        K --> Packed Kronecker set of dimensional gram matrices of the inducing points
        y --> Nx1 Target values
        noise --> scalar that represents the noise
        tolerance --> Stopping criteria of residual
        maxiter --> Maximum number of iterations
          
    Outputs:
        alpha --> Nx1 solution to linear equation
        ind  --> indicator of successful convergence. 0 if converged, else maxiter.
    '''    
   
    # initialize alpha
    alpha = np.zeros((len(y),1))
    # Perform first iteration 
    r_last = y
    p = r_last
    norm_r_last = np.linalg.norm(r_last)**2
    
    mvm = kron_MVM(W,K,p,noise)
    a = norm_r_last/np.dot(p.T,mvm)
    alpha = alpha + a*p
    r = r_last-a*mvm
    norm_r = np.linalg.norm(r)**2 
    iter = 1
    
    
    while (iter <maxiter and norm_r**0.5>tolerance):
        
        r_last=r
        B = norm_r/norm_r_last
        p = r_last+B*p
        norm_r_last = norm_r
        
        mvm = kron_MVM(W,K,p,noise)
        a = norm_r_last/np.dot(p.T,mvm)
        alpha = alpha + a*p
        r = r_last-a*mvm  
        norm_r = np.linalg.norm(r)**2 
        iter+=1
    
    #get indicator flag
    ind = 0 if iter != maxiter else iter     
    return (alpha,ind)

def minimize(X,x,y, maxnumlinesearch=20, maxnumfuneval=None, red=1.0, verbose=False):
    INT = 0.1;# don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0;              # extrapolate maximum 3 times the current step-size
    MAX = 20;                     # max 20 function evaluations per line search
    RATIO = 10;                                   # maximum allowed slope ratio
    SIG = 0.1;RHO = SIG/2;# SIG and RHO are the constants controlling the Wolfe-
    #Powell conditions. SIG is the maximum allowed absolute ratio between
    #previous and new slopes (derivatives in the search direction), thus setting
    #SIG to low (positive) values forces higher precision in the line-searches.
    #RHO is the minimum allowed fraction of the expected (from the slope at the
    #initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    #Tuning of SIG (depending on the nature of the function to be optimized) may
    #speed up the minimization; it is probably not worth playing much with RHO.

    SMALL = 10.**-16                    #minimize.m uses matlab's realmin 
    
    if maxnumlinesearch == None:
        if maxnumfuneval == None:
            raise "Specify maxnumlinesearch or maxnumfuneval"
        else:
            S = 'Function evaluation'
            length = maxnumfuneval
    else:
        if maxnumfuneval != None:
            raise "Specify either maxnumlinesearch or maxnumfuneval (not both)"
        else: 
            S = 'Linesearch'
            length = maxnumlinesearch

    i = 0                                         # zero the run length counter
    ls_failed = 0                          # no previous line search has failed                        
    df0,f0  = kernels.Derivative_Gaussian(x,y,X)  
    fX = [f0]
    i = i + (length<0)                                         # count epochs?!
    s = -df0; d0 = -dot(s.T,s)[0,0]    # initial search direction (steepest) and slope
    x3 = red/(1.0-d0)                             # initial step is red/(|s|+1)

    while i < abs(length):                                 # while not finished
        i = i + (length>0)                                 # count iterations?!

        X0 = X; F0 = f0; dF0 = df0              # make a copy of current values
        if length>0:
            M = MAX
        else: 
            M = min(MAX, -length-i)
        while 1:                      # keep extrapolating as long as necessary
            x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0
            success = 0
            while (not success) and (M > 0):
                try:
                    M = M - 1; i = i + (length<0)              # count epochs?!
                    df3,f3 = kernels.Derivative_Gaussian(x,y,X+x3*s)
                    if isnan(f3) or isinf(f3) or any(isnan(df3)+isinf(df3)):
                        if verbose : print ("error")
                        X = X - x3*s
                        return X , fX, -1
                    success = 1
                except:                    # catch any error which occured in f
                    x3 = (x2+x3)/2                       # bisect and try again
            if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3   # keep best values
            d3 = dot(df3.T,s)[0,0]  
                                   # new slope
            if d3 > SIG*d0 or f3 > f0+x3*RHO*d0 or M == 0:                                   # are we done extrapolating?
                break
            x1 = x2; f1 = f2; d1 = d2                 # move point 2 to point 1
            x2 = x3; f2 = f3; d2 = d3                 # move point 3 to point 2
            A = 6*(f1-f2)+3*(d2+d1)*(x2-x1)          # make cubic extrapolation
            B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)
            Z = B+sqrt(complex(B*B-A*d1*(x2-x1)))
            if Z != 0.0:
                x3 = x1-d1*(x2-x1)**2/Z              # num. error possible, ok!
            else: 
                x3 = inf
            if (not isreal(x3)) or isnan(x3) or isinf(x3) or (x3 < 0): 
                                                       # num prob | wrong sign?
                x3 = x2*EXT                        # extrapolate maximum amount
            elif x3 > x2*EXT:           # new point beyond extrapolation limit?
                x3 = x2*EXT                        # extrapolate maximum amount
            elif x3 < x2+INT*(x2-x1):  # new point too close to previous point?
                x3 = x2+INT*(x2-x1)
            x3 = real(x3)

        while (abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0:  
                                                           # keep interpolating
            if (d3 > 0) or (f3 > f0+x3*RHO*d0):            # choose subinterval
                x4 = x3; f4 = f3; d4 = d3             # move point 3 to point 4
            else:
                x2 = x3; f2 = f3; d2 = d3             # move point 3 to point 2
            if f4 > f0:           
                x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2))
                                                      # quadratic interpolation
            else:
                A = 6*(f2-f4)/(x4-x2)+3*(d4+d2)           # cubic interpolation
                B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
                if A != 0:
                    x3=x2+(sqrt(B*B-A*d2*(x4-x2)**2)-B)/A
                                                     # num. error possible, ok!
                else:
                    x3 = inf
            if isnan(x3) or isinf(x3):
                x3 = (x2+x4)/2      # if we had a numerical problem then bisect
            x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2))  
                                                       # don't accept too close
            df3,f3 = kernels.Derivative_Gaussian(x,y,X+x3*s)
            if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3              # keep best values
            M = M - 1; i = i + (length<0)                      # count epochs?!
            d3 = dot(df3.T,s)[0,0]                                         # new slope

        if abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0:  # if line search succeeded
            X = X+x3*s; f0 = f3; fX.append(f0)               # update variables
            if verbose: print ('%s %6i;  Value %4.6e\r' % (S, i, f0))
            s = (dot(df3.T,df3)-dot(df0.T,df3))/dot(df0.T,df0)*s - df3
                                                  # Polack-Ribiere CG direction
            df0 = df3                                        # swap derivatives
            d3 = d0; d0 = dot(df0.T,s) [0,0]
            if d0 > 0:                             # new slope must be negative
                s = -df0; d0 = -dot(s.T,s)     # otherwise use steepest direction
            x3 = x3 * min(RATIO, d3/(d0-SMALL))     # slope ratio but max RATIO
            ls_failed = 0                       # this line search did not fail
        else:
            X = X0; f0 = F0; df0 = dF0              # restore best point so far
            if ls_failed or (i>abs(length)):# line search failed twice in a row
                break                    # or we ran out of time, so we give up
            s = -df0; d0 = -dot(s.T,s)[0,0]                             # try steepest
            x3 = 1/(1-d0)                     
            ls_failed = 1                             # this line search failed
    if verbose: print (" \n")
    return X, fX, i   

def minimize_kron(X,W,x,y, maxnumlinesearch=100, maxnumfuneval=None, red=1.0, verbose=True):
    INT = 0.1;# don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0;              # extrapolate maximum 3 times the current step-size
    MAX = 20;                     # max 20 function evaluations per line search
    RATIO = 10;                                   # maximum allowed slope ratio
    SIG = 0.1;RHO = SIG/2;# SIG and RHO are the constants controlling the Wolfe-
    #Powell conditions. SIG is the maximum allowed absolute ratio between
    #previous and new slopes (derivatives in the search direction), thus setting
    #SIG to low (positive) values forces higher precision in the line-searches.
    #RHO is the minimum allowed fraction of the expected (from the slope at the
    #initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    #Tuning of SIG (depending on the nature of the function to be optimized) may
    #speed up the minimization; it is probably not worth playing much with RHO.

    SMALL = 10.**-16                    #minimize.m uses matlab's realmin 
    
    if maxnumlinesearch == None:
        if maxnumfuneval == None:
            raise "Specify maxnumlinesearch or maxnumfuneval"
        else:
            S = 'Function evaluation'
            length = maxnumfuneval
    else:
        if maxnumfuneval != None:
            raise "Specify either maxnumlinesearch or maxnumfuneval (not both)"
        else: 
            S = 'Linesearch'
            length = maxnumlinesearch

    i = 0                                         # zero the run length counter
    ls_failed = 0                          # no previous line search has failed                        
    df0,f0  = kernels.Derivative_Gaussian_Kron(W,x,y,X)  
    fX = [f0]
    i = i + (length<0)                                         # count epochs?!
    s = -df0; d0 = -dot(s.T,s)[0,0]    # initial search direction (steepest) and slope
    x3 = red/(1.0-d0)                             # initial step is red/(|s|+1)

    while i < abs(length):                                 # while not finished
        i = i + (length>0)                                 # count iterations?!

        X0 = X; F0 = f0; dF0 = df0              # make a copy of current values
        if length>0:
            M = MAX
        else: 
            M = min(MAX, -length-i)
        while 1:                      # keep extrapolating as long as necessary
            x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0
            success = 0
            while (not success) and (M > 0):
                try:
                    M = M - 1; i = i + (length<0)              # count epochs?!
                    df3,f3 = kernels.Derivative_Gaussian_Kron(W,x,y,1.0*(X+x3*s))
                    if isnan(f3) or isinf(f3) or any(isnan(df3)+isinf(df3)):
                        if verbose : print ("error")
                        X = X - x3*s
                        return X , fX, -1
                    success = 1
                except:                    # catch any error which occured in f
                    x3 = (x2+x3)/2                       # bisect and try again
            if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3   # keep best values
                print(X0)
            d3 = dot(df3.T,s)[0,0]  
                                   # new slope
            if d3 > SIG*d0 or f3 > f0+x3*RHO*d0 or M == 0:                                   # are we done extrapolating?
                break
            x1 = x2; f1 = f2; d1 = d2                 # move point 2 to point 1
            x2 = x3; f2 = f3; d2 = d3                 # move point 3 to point 2
            A = 6*(f1-f2)+3*(d2+d1)*(x2-x1)          # make cubic extrapolation
            B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)
            Z = B+sqrt(complex(B*B-A*d1*(x2-x1)))
            if Z != 0.0:
                x3 = x1-d1*(x2-x1)**2/Z              # num. error possible, ok!
            else: 
                x3 = inf
            if (not isreal(x3)) or isnan(x3) or isinf(x3) or (x3 < 0): 
                                                       # num prob | wrong sign?
                x3 = x2*EXT                        # extrapolate maximum amount
            elif x3 > x2*EXT:           # new point beyond extrapolation limit?
                x3 = x2*EXT                        # extrapolate maximum amount
            elif x3 < x2+INT*(x2-x1):  # new point too close to previous point?
                x3 = x2+INT*(x2-x1)
            x3 = real(x3)

        while (abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0:  
                                                           # keep interpolating
            if (d3 > 0) or (f3 > f0+x3*RHO*d0):            # choose subinterval
                x4 = x3; f4 = f3; d4 = d3             # move point 3 to point 4
            else:
                x2 = x3; f2 = f3; d2 = d3             # move point 3 to point 2
            if f4 > f0:           
                x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2))
                                                      # quadratic interpolation
            else:
                A = 6*(f2-f4)/(x4-x2)+3*(d4+d2)           # cubic interpolation
                B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
                if A != 0:
                    x3=x2+(sqrt(B*B-A*d2*(x4-x2)**2)-B)/A
                                                     # num. error possible, ok!
                else:
                    x3 = inf
            if isnan(x3) or isinf(x3):
                x3 = (x2+x4)/2      # if we had a numerical problem then bisect
            x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2))  
                                                       # don't accept too close
            df3,f3 = kernels.Derivative_Gaussian_Kron(W,x,y,1.0*(X+x3*s))
            if f3 < F0:
                X0 = X+x3*s; F0 = f3; dF0 = df3              # keep best values
            M = M - 1; i = i + (length<0)                      # count epochs?!
            d3 = dot(df3.T,s)[0,0]                                         # new slope

        if abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0:  # if line search succeeded
            X = X+x3*s; f0 = f3; fX.append(f0)               # update variables
            if verbose: print ('%s %6i;  Value %4.6e\r' % (S, i, f0))
            s = (dot(df3.T,df3)-dot(df0.T,df3))/dot(df0.T,df0)*s - df3
                                                  # Polack-Ribiere CG direction
            df0 = df3                                        # swap derivatives
            d3 = d0; d0 = dot(df0.T,s) [0,0]
            if d0 > 0:                             # new slope must be negative
                s = -df0; d0 = -dot(s.T,s)     # otherwise use steepest direction
            x3 = x3 * min(RATIO, d3/(d0-SMALL))     # slope ratio but max RATIO
            ls_failed = 0                       # this line search did not fail
        else:
            X = X0; f0 = F0; df0 = dF0              # restore best point so far
            if ls_failed or (i>abs(length)):# line search failed twice in a row
                break                    # or we ran out of time, so we give up
            s = -df0; d0 = -dot(s.T,s)[0,0]                             # try steepest
            x3 = 1/(1-d0)                     
            ls_failed = 1                             # this line search failed
    if verbose: print (" \n")
    return X, fX, i   

    
if __name__ == '__main__':       
    data = np.load('Regression_data.npz')
    x = data['x']
    y = data['y']
    params = np.array([[3],[3],[3]])
    params, ML, i = minimize(params,x,y,maxnumlinesearch=100)
    
    
    
                             
    