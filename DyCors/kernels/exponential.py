import numpy as np
import scipy.linalg as la

def surrogateRBF_Expo(x, f, l=None):
    """Build RBF surrogate model using Exponential kernel.
    
    Parameters
    ----------
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    f : ndarray, shape (m,)
        Array of function values at ``x``.
    l : ndarray, shape (d,), optional
        Array with the values of the internal parameter of the kernel.
        
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
    
    if l is None:
        l = np.ones(d)
    
    # RBF-matrix
    R   = -2*np.dot(x/l, x.T/l[:,np.newaxis]) + np.sum(x**2/l**2, axis=1) \
        + np.sum(x.T**2/l[:,np.newaxis]**2, axis=0)[:,np.newaxis]
    Phi = np.exp(-R/2)

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

def evalRBF_Expo(x, s, y, l=None):
    """Evaluate RBF Exponential kernel surrogate model at new points.
    
    Parameters
    ----------
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    s : ndarray, shape(m,)
        RBF coefficients.
    y : ndarray, shape (n,d,)
        Array of points where we want to evaluate the surrogate model.
    l : ndarray, shape (d,), optional
        Array with the values of the internal parameter of the kernel.
    
    Returns
    -------
    f : ndarray, shape(n,)
        Array of interpolated values.
    """
    y = np.array(y)
    n,d = y.shape
    
    if l is None:
        l = np.ones(d)

    # RBF-matrix
    R   = -2*np.dot(x/l, y.T/l[:,np.newaxis]) + np.sum(y**2/l**2, axis=1) \
        + np.sum(x**2/l**2, axis=1)[:,np.newaxis]
    Phi = np.exp(-R.T/2)

    # polynomial part
    P = np.hstack((np.ones((n,1)),y))
    A = np.block([Phi,P])
    
    # evaluation
    f = np.dot(A, s)
    
    return f

def surrogateGRBF_Expo(x, f, df, l=None):
    """Build GRBF surrogate model using Exponential kernel.
    
    Parameters
    ----------
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    f : ndarray, shape (m,)
        Array of function values at ``x``.
    df : ndarray, shape (m,d,)
        Array of function gradient values at ``x``.
    l : ndarray, shape (d,), optional
        Array with the values of the internal parameter of the kernel.
        
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
    
    if l is None:
        l = np.ones(d)

    # RBF-matrix
    R   = -2*np.dot(x/l, x.T/l[:,np.newaxis]) + np.sum(x**2/l**2, axis=1) \
        + np.sum(x.T**2/l[:,np.newaxis]**2, axis=0)[:,np.newaxis]
    Phi = np.exp(-R/2) 
    
    # First derivative
    _Phi_d = np.zeros((m,m,d))
    _Phi_d = -2*Phi[...,np.newaxis] * (x[:,np.newaxis,:] - x[np.newaxis,:,:])\
        / (2*l[np.newaxis,np.newaxis,:]**2)
    Phi_d = _Phi_d.reshape((m,m*d))

    # Second derivative
    Phi_dd = np.zeros((m,d,m,d))
    Phi_dd = - 2*_Phi_d[:,np.newaxis,:,:] \
        * (x[:,:,np.newaxis,np.newaxis] - x.T[np.newaxis,:,:,np.newaxis]) \
        / (2*l[np.newaxis,:,np.newaxis,np.newaxis]**2) \
        - np.diag(np.ones(d))[np.newaxis,:,np.newaxis,:] \
        * 2*Phi[:,np.newaxis,:,np.newaxis] \
        / (2*l[np.newaxis,:,np.newaxis,np.newaxis]**2)
    Phi_dd = Phi_dd.reshape((m*d,m*d))

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
    
def evalGRBF_Expo(x, s, y, l=None):
    """Evaluate GRBF Exponential kernel surrogate model at new points.
    
    Parameters
    ----------
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    s : ndarray, shape(m,)
        RBF coefficients.
    y : ndarray, shape (n,d,)
        Array of points where we want to evaluate the surrogate model.
    l : ndarray, shape (d,), optional
        Array with the values of the internal parameter of the kernel.
    
    Returns
    -------
    f : ndarray, shape(n,)
        Array of interpolated values.
    """
    m = x.shape[0]
    y = np.array(y)
    n,d = y.shape
    
    if l is None:
        l = np.ones(d)
    
    # RBF-matrix
    R   = -2*np.dot(x/l, y.T/l[:,np.newaxis]) + np.sum(y**2/l**2, axis=1) \
        + np.sum(x**2/l**2, axis=1)[:,np.newaxis]
    Phi = np.exp(-R.T/2)
    
    # First derivative 
    d_Phi = np.zeros((n,m,d))
    d_Phi = -2*Phi[...,np.newaxis] * (y[:,np.newaxis,:] - x[np.newaxis,:,:]) \
        / (2*l[np.newaxis,np.newaxis,:]**2)
    d_Phi = d_Phi.reshape((n,m*d))

    A = np.block([[Phi,d_Phi]])
    
    # evaluation
    f = np.dot(A,s)
    
    return f
