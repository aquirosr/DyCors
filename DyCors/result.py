import numpy as np
from scipy.optimize import OptimizeResult
import scipy.linalg as la
import matplotlib.pyplot as plt

class ResultDyCors(OptimizeResult):
    """Represents the optimization result.
    
    Inherits from `scipy.optimize.OptimizeResult
    <https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.optimize.OptimizeResult.html>`_.
    
    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Refer to `message` for
        details.
    message : str
        Description of the cause of the termination.
    fun, jac : ndarray
        Values of objective function and its Jacobian.
    nfev, njev : int
        Number of evaluations of the objective functions and of its
        Jacobian.
    nit : int
        Number of restarts performed by DyCors.
    m : int, optional
        Number of initial sampling points
    hist : ndarray, optional
        Values of objective function at all iterations.
    dhist : ndarray, optional
        Values of gradient at all iterations.
    xres : ndarray, optional
        Array with all the points evaluated by the optimization
        algorithm.
    fres : ndarray, optional
        Array with the values of the objective function at all points
        that have been evaluated.
    gres : ndarray, optional
        Array with the values of the gradient of the objective function
        at all points that have been evaluated.
    restart_its : list, optional
        List with iterations at which the optimization algorithm has
        been restarted
    """
    def __init__(self, fun, jac, nfev, njev, nit, status,
                 message, x, success, m=None, hist=None,
                 dhist=None, xres=None, fres=None, gres=None,
                 restart_its=None):
        super().__init__({"fun":fun, "jac":jac, "nfev":nfev,
                          "njev":njev, "nit":nit, "status":status,
                          "message":message, "x":x, "success":success})
        self.scipy_dict = self.copy()
        
        self["m"] = m
        self["hist"] = None
        if hist is not None and m is not None:
            self["hist"] = np.arange(m, nfev+1), hist[m-1:]
        
        self["dhist"] = None
        if dhist is not None and m is not None:
            self["dhist"] = np.arange(m, nfev+1), la.norm(dhist[m-1:,:], axis=-1)
        
        self["xres"] = xres
        self["fres"] = fres
        self["gres"] = gres
        
        self["restart_its"] = restart_its
        
    def __repr__(self):
        if self.scipy_dict.keys():
            m = max(map(len, list(self.scipy_dict.keys()))) + 1
            return "\n".join([k.rjust(m) + ": " + repr(v)
                              for k, v in sorted(self.scipy_dict.items())])
        else:
            return self.__class__.__name__ + "()"
    
    def plot(self, figsize=(), ylim_f=(), ylim_df=(), fontsize=10):
        """Plot evolution of minimum value and norm of the gradient
        if used.
        
        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure.
        ylim_f : tuple, optional
            y limits on the function history.
        ylim_df : tuple, optional
            y limits on the function history.
        fontsize : int, optional
            Font size.
        """
        if self["hist"] is None:
            return None
        
        if figsize:
            fig, ax1 = plt.subplots(figsize=figsize)
        else:
            fig, ax1 = plt.subplots()

        im1 = ax1.plot(self["hist"][0], self["hist"][1], "C0",
                       label=r"$f(x)$")
        ax1.set_xlabel("its", fontsize=fontsize)
        ax1.set_ylabel(r"$f(x)$", fontsize=fontsize)
        ax1.tick_params(axis='both', which='major', labelsize=fontsize)
        if ylim_f:
            ax1.set_ylim(*ylim_f)
        
        if self["dhist"] is not None:
            ax2 = ax1.twinx()
            im2 = ax2.plot(self["dhist"][0], self["dhist"][1], "C1",
                           label=r"$|\mathrm{d}f(x)|$")
            ax2.set_ylabel(r"$|\mathrm{d}f(x)|$", fontsize=fontsize)
            ax2.tick_params(axis='y', which='major', labelsize=fontsize)
            if ylim_df:
                ax2.set_ylim(*ylim_df)
            
            ims = im1 + im2
            labels = [im.get_label() for im in ims]
            ax1.legend(ims, labels, fontsize=fontsize)