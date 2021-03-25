import numpy as np
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt

class ResultDyCors(OptimizeResult):
    """Represents the optimization result.
    
    Inherits from scipy.OptimizeResult. 
    
    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Refer to 
        `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac: ndarray
        Values of objective function and its Jacobian.
    nfev, njev: int
        Number of evaluations of the objective functions and of its
        Jacobian.
    nit : int
        Number of restarts performed by DyCors.
    m : int, optional
        Numbe of initial sampling points
    hist : ndarray, optional
        Values of objective function at all iterations.
        
    Methods
    ----------
    plot(figsize=(), ylim=(), fontsize=10)
        Plot evolution of minimum value.
    """
    def __init__(self, fun, jac, nfev, njev, nit, status,
                 message, x, success, m=None, hist=None):
        super().__init__({"fun":fun, "jac":jac, "nfev":nfev,
                          "njev":njev, "nit":nit, "status":status,
                          "message":message, "x":x, "success":success})
        self.scipy_dict = self.copy()
        
        self.hist = hist
        if self.hist is not None and m is not None:
            self["hist"] = np.arange(m, nfev+1), self.hist[m-1:]
        
    def __repr__(self):
        if self.scipy_dict.keys():
            m = max(map(len, list(self.scipy_dict.keys()))) + 1
            return "\n".join([k.rjust(m) + ": " + repr(v)
                              for k, v in sorted(self.scipy_dict.items())])
        else:
            return self.__class__.__name__ + "()"
    
    def plot(self, figsize=(), ylim=(), fontsize=10):
        """Plot evolution of minimum value.
        
        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure.
        ylim : tuple, optional
            y limits.
        fontsize : int, optional
            Font size.
        """
        if self.hist is None:
            return None
        
        if figsize:
            fig = plt.figure(figsize=figsize)
        
        plt.plot(self["hist"][0], self["hist"][1])
        plt.xlabel("its", fontsize=fontsize)
        plt.ylabel(r"$f(x)$", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
        if ylim:
            plt.ylim(*ylim)
