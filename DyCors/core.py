from math import log
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.optimize import differential_evolution, NonlinearConstraint
import warnings

from .kernels import RBF_Exponential, GRBF_Exponential
from .kernels import RBF_Matern, GRBF_Matern
from .kernels import RBF_Cubic, GRBF_Cubic
from .result import ResultDyCors
from .sampling import ERLatinHyperCube

EPS = np.finfo(np.float64).eps

DEFAULT_OPTIONS = {"Nmax":250, "sig0":0.2, "sigm":0.2/2**6, "Ts":3, "Tf":5,
                   "weights":[0.3,0.5,0.8,0.95], "l":None, "nu":5/2,
                   "optim_loo":False, "nits_loo":40, "warnings":True}

METHODS = ["RBF-Expo", "RBF-Matern", "RBF-Cubic",
           "GRBF-Expo", "GRBF-Matern", "GRBF-Cubic"]

def minimize(fun, x0=None, args=(), method="RBF-Cubic", jac=None, bounds=None, 
             options=None, restart=None, verbose=True):
    """Minimization of scalar function of one or more variables using
    DyCors algorithm [1]_.
    
    This function is a wrapper around the class
    :class:`DyCorsMinimize`.

    The only mandatory parameters are ``fun`` and either ``x0`` or
    ``restart``.
    
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an 1-D array with shape (d,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (m,d,), optional
        Starting sampling points. m is the number of sampling points
        and d is the number of dimensions.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun` and `jac` functions).
    method : str, optional
        Kernel function to be used. Should be:

            - 'RBF-Expo'   : derivative-free with exponential kernel
            - 'RBF-Matern' : derivative-free with Matérn kernel
            - 'RBF-Cubic'  : derivative-free with cubic kernel
            - 'GRBF-Expo'  : gradient-enhanced with exponential kernel
            - 'GRBF-Matern': gradient-enhanced with Matérn kernel
            - 'GRBF-Cubic' : gradient-enhanced with cubic kernel
        
        The default method is 'RBF-Cubic'. See :ref:`Kernel functions`
        for more details on each specific method.
    jac : callable, optional
        It should return the gradient of `fun`.

            ``jac(x, *args) -> array_like, shape (d,)``

        where ``x`` is an 1-D array with shape (d,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
        Only necessary for 'GRBF-Expo', 'GRBF-Matern' and
        'GRBF-Cubic' methods.
    bounds : ndarray, shape (d,2,), optional
        Bounds on variables. If not provided, the default is not to
        use any bounds on variables.
    options : dict, optional
        A dictionary of solver options:

            Nmax : int
                Maximum number of function evaluations in serial.
            sig0 : float or ndarray
                Initial standard deviation to create new trial points.
            sigm : float or ndarray
                Minimum standard deviation to create new trial points.
            Ts : int
                Number of consecutive successful function evaluations 
                before increasing the standard deviation to create new
                trial points.
            Tf : int
                Number of consecutive unsuccessful function evaluations
                before decreasing the standard deviation to create new
                trial points.
            weights: list
                Weights that will be used to compute the scores of the
                trial points.
            l : float or ndarray
                Kernel internal parameter. Kernel width.
            nu : float (half integer)
                Matérn kernel internal parameter. Order of the Bessel
                Function.
            optim_loo : boolean
                Whether or not to use optimization of internal
                parameters.
            nits_loo : int
                Optimize internal parameters after every ``nits_loo``
                iterations.
            warnings : boolean
                Whether or not to print solver warnings.
        
    restart : ResultDyCors, optional
        Restart optimization from a previous optimization.
    verbose : boolean, optional
        Whether or not to print information of the solver iterations.

    Returns
    -------
    res : ResultDyCors
        The optimization result represented as a ``ResultDyCors`` 
        object. Important attributes are: ``x`` the solution array,
        ``success`` a Boolean flag indicating if the optimizer exited
        successfully and ``message`` which describes the cause of the
        termination. See :class:`ResultDyCors <.result.ResultDyCors>`
        for a description of other attributes.
    
    References
    ----------
    .. [1] Regis, R G and C A Shoemaker. 2013. Combining radial basis
        function surrogates and dynamic coordinate search in
        high-dimensional expensive black-box optimization. Engineering
        Optimization 45 (5): 529-555.
    
    Examples
    --------
    Let us consider the problem of minimizing the quadratic function.
    
    .. math:: 
        f(x) = x^2
    
    >>> import numpy as np
    >>> from DyCors import minimize
    
    We define the objective function, the initial sampling points and
    the boundaries of the domain as follows:
    
    >>> fun = lambda x: x[0]**2
    >>> x0 = np.array([-2.0, 2.0])[:,np.newaxis]
    >>> bounds = np.array([-5.0, 5.0])[np.newaxis,:]
    
    Finally, we run the optimization and print the results:
    
    >>> res = minimize(fun, x0, bounds=bounds,
    ...                options={"warnings":False},
    ...                verbose=False)
    >>> print(res["x"], res["fun"])
    [1.32665389e-05] 1.7600105366604962e-10
    
    We can also restart the optimization:
    
    >>> res = minimize(fun, bounds=bounds,
    ...                options={"Nmax":500, "warnings":False},
    ...                restart=res, verbose=False)
    >>> print(res["x"], res["fun"])
    [1.55369877e-06] 2.413979870364038e-12
    
    """

    # check options are ok
    if not callable(fun):
        raise TypeError("fun is not callable")

    if x0 is not None and restart is not None:
        raise ValueError("set either x0 or restart, not both")
    elif x0 is None and restart is None:
        raise ValueError("set either x0 or restart")
    elif x0 is not None:
        if x0.ndim!=2:
            raise ValueError("dimensions of x0 are not 2")
        elif bounds is not None and x0.shape[1]!=bounds.shape[0]:
            raise ValueError("dimension mismatch. Modify x0 or bounds")
    else:
        if not isinstance(restart, ResultDyCors):
            raise TypeError("restart must be ResultDyCors, not %s"%type(restart))
        elif restart["xres"] is None or restart["fres"] is None:
            raise ValueError("restart object is not properly initialized")
        elif method.startswith("G") and restart["gres"] is None:
            raise ValueError("restart object does not have the gradient "
                             + "information")
        elif bounds is not None and restart["xres"].shape[1]!=bounds.shape[0]:
            raise ValueError("dimension mismatch. Modify bounds")

    if method not in METHODS:
        raise ValueError("invalid method %s,\n  valid options are %s"
                         %(method, METHODS))
    elif method.startswith("G"):
        if not callable(jac):
            raise TypeError("jac is not callable")

    if options is None:
        options = DEFAULT_OPTIONS
    else:
        for key in DEFAULT_OPTIONS:
            if key not in options.keys():
                options[key] = DEFAULT_OPTIONS[key]

    # ignore annoying warnings
    if not options["warnings"]:
        warnings.filterwarnings("ignore", category=sla.LinAlgWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    # run the optimization
    DyCorsMin = DyCorsMinimize(fun, x0, args, method, jac, bounds, options,
                               restart, verbose)
    return DyCorsMin()

class DyCorsMinimize:
    """Implementation of DyCors algorithm.
    
    For a full description of the different options see :func:`minimize`
    """
    def __init__(self, fun, x0, args, method, jac, bounds, options, restart,
                 verbose):
        self.fun = fun
        self.x0 = x0
        self.args = args
        self.method = method
        self.jac = jac
        self.bounds = bounds
        self.options = options
        self.restart = restart
        self.verbose = verbose
        
        self.Nmax  = self.options["Nmax"]
        if self.restart is None:
            self.m, self.d = self.x0.shape
        else:
            self.d = self.restart["xres"].shape[1]
        
        if self.bounds is not None:
            self.bL, self.bU = self.bounds[:,0], self.bounds[:,1]
            self.sigm = self.options["sigm"]*(self.bU - self.bL)
            self.sig = self.options["sig0"]*(self.bU - self.bL)  
        else:
            self.sigm = self.options["sigm"]*np.ones((self.d,))
            self.sig  = self.options["sig0"]*np.ones((self.d,))

        if self.method.startswith("G"):
            self.grad = True
        else:
            self.grad = False

        self.la = sla
            
        # compute starting points
        self.fevals = 0
        if self.restart is None:
            self.initialize()
        else:
            self.initialize_restart()
            
        if self.options["l"] is None:
            if not self.grad:
                self.l = sla.norm(self.x, axis=0)
            else:
                tmp_grad = self.df.reshape(self.m, self.d)
                tmp_ip = np.mean(tmp_grad, axis=0)
                self.l = 1/tmp_ip
        elif isinstance(self.options["l"], float):
            self.l = self.options["l"]*np.ones((self.d,))
        else:
            self.l = self.options["l"]*np.ones((self.d,))

        self.nu = self.options["nu"]
            
        if self.method=="RBF-Expo":
            self.kernel = RBF_Exponential(self.l)
        elif self.method=="RBF-Matern":
            self.kernel = RBF_Matern(self.l, self.nu)
        elif self.method=="RBF-Cubic":
            self.kernel = RBF_Cubic(self.l)
        elif self.method=="GRBF-Expo":
            self.kernel = GRBF_Exponential(self.l)
        elif self.method=="GRBF-Matern":
            self.kernel = GRBF_Matern(self.l, self.nu)
        elif self.method=="GRBF-Cubic":
            self.kernel = GRBF_Cubic(self.l)
            
        self.n0, self.Np = self.m, self.Nmax - self.m
        
        self.k = min(100*self.d, 5000) # number of trial points
        self.Ts, self.Tf = self.options["Ts"], max(self.d, self.options["Tf"])
        self.weights = self.options["weights"]
        self.optim_loo = self.options["optim_loo"]
        self.nits_loo = self.options["nits_loo"]
        
        self.xres = None
        self.fres = None
        if self.grad:
            self.gres = None
        self.restart_its = []
            
        self.iB = np.argmin(self.f) # find best solution
        self.xB, self.fB = self.x[self.iB,:], np.asarray([self.f[self.iB]])
        if self.grad:
            self.dfB = self.df[self.iB*self.d:(self.iB+1)*self.d]
        self.fBhist = [self.fB[0] for i in range(self.fevals)]
        if self.grad:
            self.dfBhist = []
            for i in range(self.fevals):
                self.dfBhist.append(self.dfB)
        
    def __call__(self):
        """Perform the optimization.
        """
        try:
            nits = 1

            if self.verbose:
                print("nits = %d"%(nits))
                
            while (self.fevals < self.Nmax):
                # update internal parameters
                if self.optim_loo and (self.ic+self.m)%self.nits_loo==0:
                    self.kernel.optimize()
                    
                # fit response surface model
                if self.grad:
                    _ = self.kernel.fit(self.x, self.f, self.df)
                else:
                    _ = self.kernel.fit(self.x, self.f)

                # generate trial points and select the bests
                self.trial_points()
                self.select_new_pts()

                # run function and gradient evaluations
                self.fnew = np.apply_along_axis(self.fun, 1, self.xnew, 
                                                *self.args)
                if self.grad:
                    self.dfnew = np.apply_along_axis(self.jac, 1, 
                                                        self.xnew, 
                                                        *self.args).flatten()
                self.update()

                self.ic += 1
                self.fevals += 1
                if (self.fevals%5 == 0 and self.verbose):
                    print("  fevals = %d | fmin = %.2e"%(self.fevals, 
                                                         self.fB[0]))
                    
                # restart if algorithm converged to a local minimum
                if self.sig[0]<=self.sigm[0] and self.fevals<self.Nmax-self.m:
                    self.restart_dycors()
                    nits += 1
                    
                    if self.verbose:
                        print("nits = %d"%(nits))

            task_str = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
            warnflag = 0

        except np.linalg.LinAlgError:
            task_str = "STOP: SINGULAR MATRIX"
            warnflag = 2
            
        if self.xres is not None:
            self.xres = np.r_[self.xres, self.x]
            self.fres = np.r_[self.fres, self.f]
            if self.grad:
                self.gres = np.r_[self.gres, self.df]
            
        
        return ResultDyCors(fun=self.fB[0],
                            jac=self.dfB if self.grad else None,
                            nfev=self.fevals,
                            njev=self.fevals if self.grad else None,
                            nit=nits, status=warnflag, message=task_str,
                            x=np.asarray(self.xB), success=(warnflag==0),
                            m=self.m, hist=np.asarray(self.fBhist),
                            dhist=(np.asarray(self.dfBhist)
                                if self.grad else None),
                            xres=self.x if self.xres is None else self.xres,
                            fres=self.f if self.fres is None else self.fres,
                            gres=(self.df if self.gres is None else self.gres)
                                if self.grad else None,
                            restart_its=self.restart_its)

    def initialize(self):
        """Compute function and gradient evaluations of initial
        sampling points.
        """
        # counters
        self.ic, self.Cs, self.Cf = 0, 0, 0
        self.fevals += self.m

        if self.bounds is not None:
            # enforce bounds
            bL = self.bL*np.ones_like(self.x0)
            bU = self.bU*np.ones_like(self.x0)
            self.x = np.clip(self.x0, bL, bU)
        else:
            self.x = self.x0.copy()

        # evaluate initial points
        self.f = np.apply_along_axis(self.fun, 1, self.x, *self.args)
        if self.grad:
            self.df = np.apply_along_axis(self.jac, 1, self.x, 
                                            *self.args).flatten()
            
    def initialize_restart(self):
        """Initialize optimization from a previous optimization.
        """
        if self.restart["restart_its"] is None:
            restart_it = 0
        else:
            restart_it = self.restart["restart_its"][-1]
        
        self.x = self.restart["xres"][restart_it:,:]
        self.f = self.restart["fres"][restart_it:]
        if self.grad:
            self.df = self.restart["gres"][restart_it*self.d:]
        
        self.m = self.restart["m"]
        self.ic = self.x.shape[0] - self.m
        self.Cs, self.Cf = 0, 0
        self.fevals += self.x.shape[0]
    
    def trial_points(self):
        """Generate trial points.
        """
        # probability for coordinates
        p  = min(20/self.d, 1)*(1-log(self.ic+1)/log(self.Np))
        
        # select coordinates to perturb
        om = np.random.rand(self.d)
        Ip = np.argwhere(om<=p).flatten()
        if (len(Ip)==0): Ip = np.random.randint(0, self.d)
        
        # mask
        M  = np.zeros((self.k,self.d),dtype=int)
        M[:,Ip] = 1
        
        # generate trial points
        yk = np.outer(self.k*[1], self.xB) \
            + M*np.random.normal(0, self.sig, (self.k,self.d))
        if self.bounds is not None:
            # enforce bounds
            bL = self.bL*np.ones_like(yk)
            bU = self.bU*np.ones_like(yk)
            self.yk = np.clip(yk,bL,bU)
        else:
            self.yk = yk.copy()

    def select_new_pts(self):
        """Evaluate trial points using the surrogate model, compute
        scores and select the new points where we want to run the
        expensive function evaluation.
        """
        n = self.x.shape[0]
        # estimate function value
        s = self.kernel.evaluate(self.yk)

        # compute RBF-score
        s1, s2 = s.min(), s.max()
        VR = (s-s1)/(s2-s1) if (abs(s2-s1)>1.e-8) else 1

        # compute DIS-score
        dis = np.zeros(self.k)
        for i in range(self.k):
            tmp1 = np.outer(n*[1], self.yk[i,:]) - self.x
            tmp2 = np.apply_along_axis(nla.norm, 0, tmp1)
            dis[i] = tmp2.min()
        d1, d2 = dis.min(), dis.max()
        VD = (d2-dis)/(d2-d1) if (abs(d2-d1)>1.e-8) else 1

        # combine the two scores
        wR = self.weights[self.ic%len(self.weights)]
        wD = 1 - wR
        V = wR*VR + wD*VD # full weight

        # select the next points (avoid singular RBF matrix)
        iB, iP, iX = np.argsort(V), 0, 0

        self.xnew = []
        while len(self.xnew)<1:
            xnew  = self.yk[iB[iP+iX],:]
            while ((xnew.tolist() in self.x.tolist()) 
                   or (xnew.tolist() in self.xnew)):
                iP  +=1
                if iP>=iB.shape[0]:
                    break
                xnew = self.yk[iB[iP],:]
            
            iX += 1
            self.xnew.append(xnew.tolist())

    def update(self):
        """Update information after every iteration.
        """
        # update counters
        Cs = 0
        if (self.fnew[0]<self.fB):
            self.xB, self.fB = self.xnew[0], [self.fnew[0]]
            if self.grad:
                self.dfB = self.dfnew.copy()
            Cs += 1
        
        self.fBhist.append(self.fB[0])
        if self.grad:
            self.dfBhist.append(self.dfB)
        if Cs>0:
            self.Cs += 1
            self.Cf  = 0
        else:
            self.Cf += 1
            self.Cs  = 0

        if (self.Cs>=self.Ts):
            self.sig *= 2
            self.Cs   = 0
        if (self.Cf>=self.Tf):
            self.sig = np.max([self.sig/2, self.sigm], axis=0)
            self.Cf  = 0

        # update information
        self.x = np.vstack((self.x, self.xnew))
        self.f = np.concatenate((self.f, self.fnew))
        if self.grad:
            self.df = np.concatenate((self.df, self.dfnew))
                
    def restart_dycors(self):
        """Restart DyCors keeping only the best point.
        """
        self.restart_its.append(self.fevals)
        if self.xres is None:
            self.xres = self.x.copy()
            self.fres = self.f.copy()
            if self.grad:
                self.gres = self.df.copy()
        else:
            self.xres = np.r_[self.xres, self.x]
            self.fres = np.r_[self.fres, self.f]
            if self.grad:
                self.gres = np.r_[self.gres, self.df]
        
        x = np.outer((self.m-1)*[1],self.bounds[:,0]) \
            + np.outer((self.m-1)*[1], self.bounds[:,1]-self.bounds[:,0]) \
                * ERLatinHyperCube((self.m-1),self.d)
                
        self.x0 = np.concatenate((np.asarray(self.xB)[np.newaxis,:], x),
                                 axis=0)

        # reset the bounds and counters
        if self.bounds is not None:
            self.sigm = self.options["sigm"]*(self.bU - self.bL)
            self.sig = self.options["sig0"]*(self.bU - self.bL)
        else:
            self.sigm = self.options["sigm"]*np.ones((self.d,))
            self.sig = self.options["sig0"]*np.ones((self.d,))
        
        self.initialize()
        
        self.iB = np.argmin(self.f) # find best solution
        self.xB, self.fB = self.x[self.iB,:], np.asarray([self.f[self.iB]])
        if self.grad:
            self.dfB = self.df[self.iB*self.d:(self.iB+1)*self.d]
        self.fBhist.extend([self.fB[0] for i in range(self.m)])
        if self.grad:
            for i in range(self.m):
                self.dfBhist.append(self.dfB)
        
