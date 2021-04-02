import numpy as np
import scipy.linalg as la
from scipy.special import factorial

EPS = np.finfo(np.float64).eps

def surrogateRBF_Matern(x, f, l=None, nu=None):
    """Build RBF surrogate model using half integer
    simplification of Matérn kernel.
    
    Parameters
    ----------
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    f : ndarray, shape (m,)
        Array of function values at ``x``.
    l : ndarray, shape (d,), optional
        Array with the values of the width internal parameter of the
        kernel.
    nu : float, optional
        Order of Bessel function of the kernel.
        
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
    if nu is None:
        nu = 5/2
    
    p = int(round(nu-1/2)+1e-8)

    # RBF-matrix
    R = -2*np.dot(x/l, x.T/l[:,np.newaxis]) + np.sum(x**2/l**2, axis=1) \
        + np.sum(x.T**2/l[:,np.newaxis]**2, axis=0)[:,np.newaxis]
    R[R<=0.0] = EPS

    Phi = factorial(p) / factorial(2*p) * np.exp(-np.sqrt((2*p+1)*R))
    tmp = np.zeros_like(Phi)
    for i in range(p+1):
        tmp += factorial(p+i) / (factorial(i)*factorial(p-i)) \
            * (2*np.sqrt((2*p+1)*R))**(p-i)
    Phi *= tmp

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

def evalRBF_Matern(x, s, y, l=None, nu=None):
    """Evaluate RBF Matérn kernel surrogate model at new points.
    
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
    nu : float, optional
        Order of Bessel function of the kernel.
    
    Returns
    -------
    f : ndarray, shape(n,)
        Array of interpolated values.
    """
    y = np.array(y)
    n,d = y.shape
    
    if l is None:
        l = np.ones(d)
    if nu is None:
        nu = 5/2

    p = int(round(nu-1/2)+1e-8)

    # RBF-matrix
    R = -2*np.dot(x/l, y.T/l[:,np.newaxis]) + np.sum(y**2/l**2, axis=1) \
        + np.sum(x**2/l**2, axis=1)[:,np.newaxis]
    R[R<=0.0] = EPS

    Phi = factorial(p) / factorial(2*p) * np.exp(-np.sqrt((2*p+1)*R.T))
    tmp = np.zeros_like(Phi)
    for i in range(p+1):
        tmp += factorial(p+i) / (factorial(i)*factorial(p-i)) \
            * (2*np.sqrt((2*p+1)*R.T))**(p-i)
    Phi *= tmp

    # polynomial part
    P = np.hstack((np.ones((n,1)),y))
    A = np.block([Phi,P])
    
    # evaluation
    f = np.dot(A,s)
    
    return f

def surrogateGRBF_Matern(x, f, df, l=None, nu=None):
    """Build GRBF surrogate model using half integer
    simplification of Matérn kernel.
    
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
        Array with the values of the width internal parameter of the
        kernel.
    nu : float, optional
        Order of Bessel function of the kernel.
        
    Returns
    -------
    s : ndarray, shape(m,)
        GRBF coefficients.
    Phi : ndarray, shape(m,m,)
        RBF matrix.
    A : ndarray, shape(m*(d+1),m*(d+1),)
        gRBF matrix with gradient terms.
    """
    m,d = x.shape
    
    if l is None:
        l = np.ones(d)
    if nu is None:
        nu = 5/2
        
    p = int(round(nu-1/2)+1e-8)

    # RBF-matrix
    R = -2*np.dot(x/l, x.T/l[:,np.newaxis]) + np.sum(x**2/l**2, axis=1) \
        + np.sum(x.T**2/l[:,np.newaxis]**2, axis=0)[:,np.newaxis]
    R[R<=0.0] = EPS # R=0.0 is indeterminate
    
    # temporary matrices
    tmp0 = np.zeros_like(R)
    tmp1 = np.zeros_like(R)
    tmp2 = np.zeros_like(R)
    for i in range(p+1):
        tmp0 += factorial(p+i) / (factorial(i) * factorial(p-i)) \
            * (2*np.sqrt((2*p+1)*R))**(p-i)
        if i<p:
            tmp1 += factorial(p+i) / (factorial(i) * factorial(p-i)) \
                * (p-i) * (2*np.sqrt((2*p+1)*R))**(p-i-1) * 2*np.sqrt(2*p+1)
        if i<p-1:
            tmp2 += factorial(p+i) / (factorial(i) * factorial(p-i)) \
                * (p-i) * (p-i-1) * (2*np.sqrt((2*p+1)*R))**(p-i-2) \
                * (2*np.sqrt(2*p+1))**2

    fp_f2p_er = factorial(p) / factorial(2*p) * np.exp(-np.sqrt((2*p+1)*R))

    Phi = fp_f2p_er * tmp0

    # First derivative
    Phi_d = np.zeros((m,m,d))
    Phi_d = (Phi[:,:,np.newaxis] * (-np.sqrt(2*p+1)) \
            + fp_f2p_er[:,:,np.newaxis] * tmp1[:,:,np.newaxis]) \
        * (x[:,np.newaxis,:] - x[np.newaxis,:,:])\
        / np.sqrt(R[:,:,np.newaxis]) / l[np.newaxis,np.newaxis,:]**2
    Phi_d = Phi_d.reshape((m,m*d))

    # Second derivative
    Phi_dd = np.zeros((m,d,m,d))
    Phi_dd = (Phi[:,np.newaxis,:,np.newaxis] * (-np.sqrt(2*p+1))**2 \
            + 2 * fp_f2p_er[:,np.newaxis,:,np.newaxis] * (-np.sqrt(2*p+1)) \
            * tmp1[:,np.newaxis,:,np.newaxis] \
            + fp_f2p_er[:,np.newaxis,:,np.newaxis] \
            * tmp2[:,np.newaxis,:,np.newaxis]) \
        * (x[:,np.newaxis,np.newaxis,:] - x[np.newaxis,np.newaxis,:,:]) \
        / np.sqrt(R[:,np.newaxis,:,np.newaxis]) \
        / l[np.newaxis,np.newaxis,np.newaxis,:]**2 \
        * (x[:,:,np.newaxis,np.newaxis] - x.T[np.newaxis,:,:,np.newaxis]) \
        / np.sqrt(R[:,np.newaxis,:,np.newaxis]) \
        / l[np.newaxis,:,np.newaxis,np.newaxis]**2 \
        + (Phi[:,np.newaxis,:,np.newaxis] * (-np.sqrt(2*p+1)) \
            + fp_f2p_er[:,np.newaxis,:,np.newaxis] \
            * tmp1[:,np.newaxis,:,np.newaxis]) \
        * (np.diag(np.ones(d))[np.newaxis,:,np.newaxis,:] \
            * (np.sqrt(R[:,np.newaxis,:,np.newaxis] \
                * l[np.newaxis,:,np.newaxis,np.newaxis]**4)) \
            - (x[:,:,np.newaxis,np.newaxis] - x.T[np.newaxis,:,:,np.newaxis])\
            * (x[:,np.newaxis,np.newaxis,:] - x[np.newaxis,np.newaxis,:,:]) \
            / np.sqrt(R[:,np.newaxis,:,np.newaxis]) \
            / l[np.newaxis,np.newaxis,np.newaxis,:]**2 \
            * l[np.newaxis,:,np.newaxis,np.newaxis]**2) \
        / (np.sqrt(R[:,np.newaxis,:,np.newaxis]) \
            * l[np.newaxis,:,np.newaxis,np.newaxis]**2)**2
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

def evalGRBF_Matern(x, s, y, l=None, nu=None):
    """Evaluate GRBF Matérn kernel surrogate model at new points.
    
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
    nu : float, optional
        Order of Bessel function of the kernel.
    
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
    if nu is None:
        nu = 5/2
        
    p = int(round(nu-1/2)+1e-8)

    # RBF-matrix
    R   = -2*np.dot(x/l, y.T/l[:,np.newaxis]) + np.sum(y**2/l**2, axis=1) \
        + np.sum(x**2/l**2, axis=1)[:,np.newaxis]
    R[R<=0.0] = EPS # R=0.0 is indeterminate

    # temporary matrices
    tmp0 = np.zeros_like(R.T)
    tmp1 = np.zeros_like(R.T)
    for i in range(p+1):
        tmp0 += factorial(p+i) / (factorial(i)*factorial(p-i)) \
            * (2*np.sqrt((2*p+1)*R.T))**(p-i)
        if i<p:
            tmp1 += factorial(p+i) / (factorial(i) * factorial(p-i)) \
                * (p-i) * (2*np.sqrt((2*p+1)*R.T))**(p-i-1) * 2*np.sqrt(2*p+1)

    fp_f2p_er = factorial(p) / factorial(2*p) * np.exp(-np.sqrt((2*p+1)*R.T))

    Phi = fp_f2p_er * tmp0

    # First derivative             
    d_Phi = np.zeros((n,m,d))
    d_Phi = (Phi[:,:,np.newaxis] * (-np.sqrt(2*p+1)) \
        + fp_f2p_er[:,:,np.newaxis] * tmp1[:,:,np.newaxis]) \
        * (y[:,np.newaxis,:]-x[np.newaxis,:,:]) \
        / np.sqrt(R.T[:,:,np.newaxis]) / l[np.newaxis,np.newaxis,:]**2
    d_Phi = d_Phi.reshape((n,m*d))

    A = np.block([[Phi,d_Phi]])
    
    # evaluation
    f = np.dot(A,s)
    
    return f