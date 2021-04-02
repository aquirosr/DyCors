import numpy as np
import scipy.linalg as la

def surrogateRBF_Cubic(x, f):
    """Build RBF surrogate model using Cubic kernel.
    
    Parameters
    ----------
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    f : ndarray, shape (m,)
        Array of function values at ``x``.
        
    Returns
    -------
    s : ndarray, shape(m,)
        RBF coefficients.
    Phi : ndarray, shape(m,m,)
        RBF matrix.
    A : ndarray, shape(m*(d+1),m*(d+1),)
        RBF matrix with linear polynomial terms.
    """
    m,d = x.shape
    
    # RBF-matrix
    R = la.norm(x[...,np.newaxis] - x.T[np.newaxis,...], axis=1)
    Phi = R**3
    # print(Phi)

    # polynomial part
    P = np.hstack((np.ones((m,1)), x))
    
    # zero matrix
    Z = np.zeros((d+1,d+1))
    A = np.block([[Phi,P],[P.T,Z]])
    
    # right-hand side
    F     = np.zeros(m+d+1)
    F[:m] = f
    
    # solution
    s = la.solve(A, F)
    
    return s, Phi, A

def evalRBF_Cubic(x, s, y):
    """Evaluate surrogate model at new point.
    
    Parameters
    ----------
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    s : ndarray, shape(m,)
        RBF coefficients.
    y : ndarray, shape (n,d,)
        Array of points where we want to evaluate the surrogate model.
    """
    y = np.array(y)
    n,d = y.shape
    
    # RBF-matrix
    R = la.norm(x[...,np.newaxis] - y.T[np.newaxis,...], axis=1)
    Phi = R.T**3

    # polynomial part
    P = np.hstack((np.ones((n,1)),y))
    A = np.block([Phi,P])
    
    # evaluation
    return np.dot(A, s)

def surrogateGRBF_Cubic(x, f, df):
    """Build GRBF surrogate model using Cubic kernel.
    
    Parameters
    ----------
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    f : ndarray, shape (m,)
        Array of function values at ``x``.
    df : ndarray, shape (m,d,)
        Array of function gradient values at ``x``.
        
    Returns
    -------
    s : ndarray, shape(m,)
        GRBF coefficients.
    Phi : ndarray, shape(m,m,)
        RBF matrix.
    A : ndarray, shape(m*(d+1),m*(d+1),)
        GRBF matrix with gradient terms.
    """
    m,d = x.shape

    # RBF-matrix
    R = la.norm(x[...,np.newaxis] - x.T[np.newaxis,...], axis=1)
    Phi = R**3
    
    # First derivative
    _Phi_d = np.zeros((m,m,d))
    _Phi_d = 3 * R[...,np.newaxis] * (x[:,np.newaxis,:] - x[np.newaxis,:,:])
    Phi_d = _Phi_d.reshape((m,m*d))

    # Second derivative
    Phi_dd = np.zeros((m,d,m,d))
    Phi_dd = 3 * ( (x[:,np.newaxis,np.newaxis,:]
                    - x[np.newaxis,np.newaxis,:,:])
                  * (x[:,:,np.newaxis,np.newaxis]
                     - x.T[np.newaxis,:,:,np.newaxis])
                  / R[:,np.newaxis,:,np.newaxis]
                  + np.diag(np.ones(d))[np.newaxis,:,np.newaxis,:]
                  * R[:,np.newaxis,:,np.newaxis] )
    Phi_dd = np.nan_to_num(Phi_dd).reshape((m*d,m*d))

    A = np.block([[Phi,Phi_d],[-np.transpose(Phi_d),Phi_dd]])

    # right-hand side
    F = np.zeros(m*(d+1))

    # function value
    F[:m] = f

    # derivative function value
    F[m:m*(d+1)] = df
    
    # solution
    s = la.solve(A, F)
    
    return s, Phi, A
    
def evalGRBF_Cubic(x, s, y):
    """Evaluate surrogate model at new point.
    
    Parameters
    ----------
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    s : ndarray, shape(m,)
        RBF coefficients.
    y : ndarray, shape (n,d,)
        Array of points where we want to evaluate the surrogate model.
    """
    m = x.shape[0]
    y = np.array(y)
    n,d = y.shape
    
    # RBF-matrix
    R = la.norm(x[...,np.newaxis] - y.T[np.newaxis,...], axis=1)
    Phi = R.T**3
    
    # First derivative 
    d_Phi = np.zeros((n,m,d))
    d_Phi = 3 * R.T[...,np.newaxis] * (y[:,np.newaxis,:] - x[np.newaxis,:,:])

    d_Phi = d_Phi.reshape((n,m*d))

    A = np.block([[Phi,d_Phi]])
    
    # evaluation
    return np.dot(A,s)
