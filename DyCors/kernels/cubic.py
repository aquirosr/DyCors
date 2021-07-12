import numpy as np
import scipy.linalg as la

class RBF_Cubic():
    """RBF Cubic kernel class.
    
    Parameters
    ----------
    l : float or ndarray, shape (d,)
        Internal parameter. Width of the kernel.
    
    Attributes
    ----------
    l : float or ndarray, shape (d,)
        Internal parameter. Width of the kernel.
    s : ndarray, shape(m+d+1,)
        RBF coefficients.
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    """
    
    def __init__(self, l=1.0):
        self.l = l
        self.s = None
        self.x = None
    
    def fit(self, x, f):
        """Build surrogate model.
        
        Parameters
        ----------
        x : ndarray, shape (m,d,)
            Array of points where function values are known. m is the
            number of sampling points and d is the number of dimensions.
        f : ndarray, shape (m,)
            Array of function values at ``x``.
            
        Returns
        -------
        Phi : ndarray, shape(m,m,)
            RBF matrix.
        A : ndarray, shape(m*(d+1),m*(d+1),)
            RBF matrix with linear polynomial terms.
        """
        self.x = x
        m,d = self.x.shape
    
        l = np.ones(d)*self.l
    
        # RBF-matrix
        R = la.norm(self.x[...,np.newaxis]/l[:,np.newaxis]
                    - self.x.T[np.newaxis,...]/l[:,np.newaxis], axis=1)
        Phi = R**3

        # polynomial part
        P = np.hstack((np.ones((m,1)), self.x))
        
        # zero matrix
        Z = np.zeros((d+1,d+1))
        A = np.block([[Phi,P],[P.T,Z]])
        
        # right-hand side
        F = np.zeros(m+d+1)
        F[:m] = f
        
        # solution
        self.s = la.solve(A, F)
        
        return Phi, A
        
    def evaluate(self, y):
        """Evaluate surrogate model at given points.

        Parameters
        ----------
        y : ndarray, shape (n,d,)
            Array of points where we want to evaluate the surrogate model.

        Returns
        -------
        f : ndarray, shape(n,)
            Array of interpolated values.
        """
        if self.s is None:
            return None
        
        y = np.array(y)
        n,d = y.shape
        
        l = np.ones(d)*self.l
        
        # RBF-matrix
        R = la.norm(self.x[...,np.newaxis]/l[:,np.newaxis] 
                    - y.T[np.newaxis,...]/l[:,np.newaxis], axis=1)
        Phi = R.T**3

        # polynomial part
        P = np.hstack((np.ones((n,1)),y))
        A = np.block([Phi,P])
        
        # evaluation
        f = np.dot(A, self.s)
        return f
    
    def update(self, l=1.0):
        """Update internal parameters of the kernel.
        
        Parameters
        ----------
        l : float or ndarray, shape (d,), optional
            Internal parameter. Width of the kernel.
        """
        self.l = l
        
class GRBF_Cubic():
    """GRBF Cubic kernel class.
    
    Parameters
    ----------
    l : float or ndarray, shape (d,)
        Internal parameter. Width of the kernel.
    
    Attributes
    ----------
    l : float or ndarray, shape (d,)
        Internal parameter. Width of the kernel.
    s : ndarray, shape(m*(d+1),)
        GRBF coefficients.
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    """
    
    def __init__(self, l=1.0):
        self.l = l
        self.s = None
        self.x = None
    
    def fit(self, x, f, df):
        """Build surrogate model.
        
        Parameters
        ----------
        x : ndarray, shape (m,d,)
            Array of points where function values are known. m is the
            number of sampling points and d is the number of dimensions.
        f : ndarray, shape (m,)
            Array of function values at ``x``.
        df : ndarray, shape (m*d,)
            Array of function gradient values at ``x``.
            
        Returns
        -------
        Phi : ndarray, shape(m,m,)
            RBF matrix.
        A : ndarray, shape(m*(d+1),m*(d+1),)
            GRBF matrix with gradient terms.
        """
        self.x = x
        m,d = self.x.shape
        
        l = np.ones(d)*self.l

        # RBF-matrix
        R = la.norm(self.x[...,np.newaxis]/l[:,np.newaxis] 
                    - self.x.T[np.newaxis,...]/l[:,np.newaxis], axis=1)
        Phi = R**3
        
        # First derivative
        _Phi_d = np.zeros((m,m,d))
        _Phi_d = 3 * R[...,np.newaxis] * (self.x[:,np.newaxis,:] 
                                          - self.x[np.newaxis,:,:]) \
            / l[np.newaxis,np.newaxis,:]**2
                
        Phi_d = _Phi_d.reshape((m,m*d))

        # Second derivative
        Phi_dd = np.zeros((m,d,m,d))
        Phi_dd = 3 * ( (self.x[:,np.newaxis,np.newaxis,:]
                        - self.x[np.newaxis,np.newaxis,:,:])
                    * (self.x[:,:,np.newaxis,np.newaxis]
                        - self.x.T[np.newaxis,:,:,np.newaxis])
                    / R[:,np.newaxis,:,np.newaxis]
                    / l[np.newaxis,:,np.newaxis,np.newaxis]**2
                    / l[np.newaxis,np.newaxis,np.newaxis,:]**2
                    + np.diag(np.ones(d))[np.newaxis,:,np.newaxis,:]
                    * R[:,np.newaxis,:,np.newaxis] 
                    / l[np.newaxis,:,np.newaxis,np.newaxis]**2 )
        Phi_dd = np.nan_to_num(Phi_dd).reshape((m*d,m*d))

        A = np.block([[Phi,Phi_d],[-np.transpose(Phi_d),Phi_dd]])

        # right-hand side
        F = np.zeros(m*(d+1))

        # function value
        F[:m] = f

        # derivative function value
        F[m:m*(d+1)] = df
        
        # solution
        self.s = la.solve(A, F)
        
        return Phi, A
        
    def evaluate(self, y):
        """Evaluate surrogate model at given points.
        
        Parameters
        ----------
        y : ndarray, shape (n,d,)
            Array of points where we want to evaluate the surrogate model.
        
        Returns
        -------
        f : ndarray, shape(n,)
            Array of interpolated values.
        """
        if self.s is None:
            return None
        
        m = self.x.shape[0]
        y = np.array(y)
        n,d = y.shape
        
        l = np.ones(d)*self.l
        
        # RBF-matrix
        R = la.norm(self.x[...,np.newaxis]/l[:,np.newaxis] 
                    - y.T[np.newaxis,...]/l[:,np.newaxis], axis=1)
        Phi = R.T**3
        
        # First derivative 
        d_Phi = np.zeros((n,m,d))
        d_Phi = 3 * R.T[...,np.newaxis] * (y[:,np.newaxis,:] 
                                           - self.x[np.newaxis,:,:]) \
            / l[np.newaxis,np.newaxis,:]**2

        d_Phi = d_Phi.reshape((n,m*d))

        A = np.block([[Phi,d_Phi]])
        
        # evaluation
        f  =np.dot(A, self.s)
        
        return f

    def update(self, l=1.0):
        """Update internal parameters of the kernel.
        
        Parameters
        ----------
        l : float or ndarray, shape (d,), optional
            Internal parameter. Width of the kernel.
        """
        self.l = l