import numpy as np
import scipy.linalg as la
from scipy.special import factorial

EPS = np.finfo(np.float64).eps

class RBF_Matern():
    """RBF Matérn kernel class.
    
    Parameters
    ----------
    l : float or ndarray, shape (d,), optional
        Internal parameter. Width of the kernel.
    nu : float, optional
        Internal parameter. Order of Bessel function of the kernel.
    
    Attributes
    ----------
    l : float or ndarray, shape (d,)
        Internal parameter. Width of the kernel.
    nu : float
        Internal parameter. Order of Bessel function of the kernel.
    s : ndarray, shape(m,)
        RBF coefficients.
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    """
    
    def __init__(self, l=1.0, nu=5/2):
        self.l = l
        self.nu = nu
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
        m,d = x.shape
        
        l = np.ones(d)*self.l
        p = int(round(self.nu-1/2)+1e-8)

        # RBF-matrix
        R = -2*np.dot(self.x/l, self.x.T/l[:,np.newaxis]) \
            + np.sum(self.x**2/l**2, axis=1) \
            + np.sum(self.x.T**2/l[:,np.newaxis]**2, axis=0)[:,np.newaxis]
        R[R<=0.0] = EPS

        Phi = factorial(p) / factorial(2*p) * np.exp(-np.sqrt((2*p+1)*R))
        tmp = np.zeros_like(Phi)
        for i in range(p+1):
            tmp += factorial(p+i) / (factorial(i)*factorial(p-i)) \
                * (2*np.sqrt((2*p+1)*R))**(p-i)
        Phi *= tmp

        # polynomial part
        P = np.hstack((np.ones((m,1)), self.x))
        
        # zero matrix
        Z = np.zeros((d+1,d+1))
        A = np.block([[Phi,P],[P.T,Z]])

        # right-hand side
        F     = np.zeros(m+d+1)
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
        p = int(round(self.nu-1/2)+1e-8)

        # RBF-matrix
        R = -2*np.dot(self.x/l, y.T/l[:,np.newaxis]) \
            + np.sum(y**2/l**2, axis=1) \
            + np.sum(self.x**2/l**2, axis=1)[:,np.newaxis]
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
        f = np.dot(A, self.s)
        
        return f
    
    def update(self, l=1.0, nu=5/2):
        """Update internal parameters of the kernel.
        
        Parameters
        ----------
        l : float or ndarray, shape (d,), optional
            Internal parameter of the kernel.
        nu : float, optional
            Internal parameter. Order of Bessel function of the kernel.
        """
        self.l = l
        self.nu = nu
        
class GRBF_Matern():
    """GRBF Matérn kernel class.
    
    Parameters
    ----------
    l : float or ndarray, shape (d,), optional
        Internal parameter of the kernel.
    nu : float, optional
            Internal parameter. Order of Bessel function of the kernel.
    
    Attributes
    ----------
    l : float or ndarray, shape (d,)
        Internal parameter of the kernel.
    nu : float
        Internal parameter. Order of Bessel function of the kernel.
    s : ndarray, shape(m,)
        GRBF coefficients.
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    """
    
    def __init__(self, l=1.0, nu=5/2):
        self.l = l
        self.nu = nu
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
        df : ndarray, shape (m,d,)
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
        p = int(round(self.nu-1/2)+1e-8)

        # RBF-matrix
        R = -2*np.dot(self.x/l, self.x.T/l[:,np.newaxis]) \
            + np.sum(self.x**2/l**2, axis=1) \
            + np.sum(self.x.T**2/l[:,np.newaxis]**2, axis=0)[:,np.newaxis]
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
            * (self.x[:,np.newaxis,:] - self.x[np.newaxis,:,:])\
            / np.sqrt(R[:,:,np.newaxis]) / l[np.newaxis,np.newaxis,:]**2
        Phi_d = Phi_d.reshape((m,m*d))

        # Second derivative
        Phi_dd = np.zeros((m,d,m,d))
        Phi_dd = (Phi[:,np.newaxis,:,np.newaxis] * (-np.sqrt(2*p+1))**2 \
                + 2 * fp_f2p_er[:,np.newaxis,:,np.newaxis] * (-np.sqrt(2*p+1)) \
                * tmp1[:,np.newaxis,:,np.newaxis] \
                + fp_f2p_er[:,np.newaxis,:,np.newaxis] \
                * tmp2[:,np.newaxis,:,np.newaxis]) \
            * (self.x[:,np.newaxis,np.newaxis,:] 
               - self.x[np.newaxis,np.newaxis,:,:]) \
            / np.sqrt(R[:,np.newaxis,:,np.newaxis]) \
            / l[np.newaxis,np.newaxis,np.newaxis,:]**2 \
            * (self.x[:,:,np.newaxis,np.newaxis] 
               - self.x.T[np.newaxis,:,:,np.newaxis]) \
            / np.sqrt(R[:,np.newaxis,:,np.newaxis]) \
            / l[np.newaxis,:,np.newaxis,np.newaxis]**2 \
            + (Phi[:,np.newaxis,:,np.newaxis] * (-np.sqrt(2*p+1)) \
                + fp_f2p_er[:,np.newaxis,:,np.newaxis] \
                * tmp1[:,np.newaxis,:,np.newaxis]) \
            * (np.diag(np.ones(d))[np.newaxis,:,np.newaxis,:] \
                * (np.sqrt(R[:,np.newaxis,:,np.newaxis] \
                    * l[np.newaxis,:,np.newaxis,np.newaxis]**4)) \
                - (self.x[:,:,np.newaxis,np.newaxis] 
                   - self.x.T[np.newaxis,:,:,np.newaxis])\
                * (self.x[:,np.newaxis,np.newaxis,:] 
                   - self.x[np.newaxis,np.newaxis,:,:]) \
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
        p = int(round(self.nu-1/2)+1e-8)

        # RBF-matrix
        R   = -2*np.dot(self.x/l, y.T/l[:,np.newaxis]) \
            + np.sum(y**2/l**2, axis=1) \
            + np.sum(self.x**2/l**2, axis=1)[:,np.newaxis]
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
            * (y[:,np.newaxis,:]-self.x[np.newaxis,:,:]) \
            / np.sqrt(R.T[:,:,np.newaxis]) / l[np.newaxis,np.newaxis,:]**2
        d_Phi = d_Phi.reshape((n,m*d))

        A = np.block([[Phi,d_Phi]])
    
        # evaluation
        f = np.dot(A, self.s)
        
        return f
    
    def update(self, l=1.0, nu=5/2):
        """Update internal parameters of the kernel.
        
        Parameters
        ----------
        l : float or ndarray, shape (d,)
            Internal parameter of the kernel.
        nu : float, optional
            Internal parameter. Order of Bessel function of the kernel.
        """
        self.l = l
        self.nu = nu
