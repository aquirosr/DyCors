import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
from scipy.optimize import differential_evolution, NonlinearConstraint

EPS = np.finfo(np.float64).eps

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
    f : ndarray, shape (m,)
        Array of function values at ``x``.
    """
    
    def __init__(self, l=1.0):
        self.l = l
        self.s = None
        self.x = None
        self.f = None
    
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
        self.f = f
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
        F[:m] = self.f
        
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
    
    def constr(self, l):
        """Constraint on the condition number of the kernel matrix.
        """
        l0 = self.l.copy()
        
        self.l = l
        Phi,_ = self.fit(self.x, self.f)

        self.l = l0
        
        return nla.cond(Phi)
    
    def eloo(self, l):
        """Compute leave-one-one error.
        """
        l0 = self.l.copy()
        
        n = self.x.shape[0]
        
        self.l = l
        Phi,_ = self.fit(self.x, self.f)
        
        self.l = l0
        
        iH = la.inv(Phi)
        iH2 = iH@iH

        return nla.norm(self.f.T@iH2@self.f/(n*np.diag(iH2)),
                        ord=1)
    
    def optimize(self):
        """Optimize internal parameters based on the Leave-One-Out
        error using a differential evolution algorithm [1]_, [2]_. A
        constraint is applied to the condition number of the kernel
        matrix to ensure the smoothness of the function.
        
        References
        ----------
        .. [1] Rippa, S. 1999. An algorithm for selecting a good value
            for the parameter c in radial basis function interpolation.
            Advances in Computational Mathematics 11 (2). 193-210.
        .. [2] Bompard, M, J Peter and J A Desideri. 2010. Surrogate
            models based on function and derivative values for aerodynamic
            global optimization. V European Conference on Computational
            Fluid Dynamics ECCOMAS CFD 2010, ECCOMAS. Lisbon, Portugal.
        """
        nlc = NonlinearConstraint(self.constr, 0, 0.1/EPS)
        
        error0 = self.eloo(self.l)
        try:
            sol = differential_evolution(func=self.eloo,
                                         bounds=(((1e-5,1e1),)*self.x.shape[1]),
                                         constraints=(nlc),
                                         strategy="rand2bin", maxiter=100)
            if sol["fun"]<error0:
                self.update(l=sol["x"])
        except np.linalg.LinAlgError:
            pass
    
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
    f : ndarray, shape (m,)
        Array of function values at ``x``.
    df : ndarray, shape (m*d,)
        Array of function gradient values at ``x``.
    """
    
    def __init__(self, l=1.0):
        self.l = l
        self.s = None
        self.x = None
        self.f = None
        self.df = None
    
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
        self.f = f
        self.df = df
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
        F[:m] = self.f

        # derivative function value
        F[m:m*(d+1)] = self.df
        
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
    
    def optimize(self):
        """Optimize internal parameters as the inverse of the average
        gradient.
        """
        m,d = self.x.shape
        tmp_grad = self.df.reshape(m, d)
        tmp_ip = np.mean(tmp_grad, axis=0)
        self.l = 1/tmp_ip

    def update(self, l=1.0):
        """Update internal parameters of the kernel.
        
        Parameters
        ----------
        l : float or ndarray, shape (d,), optional
            Internal parameter. Width of the kernel.
        """
        self.l = l