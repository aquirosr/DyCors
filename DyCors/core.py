from math import log
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.optimize import differential_evolution, NonlinearConstraint
import multiprocessing
import warnings

from .kernels import RBF_Exponential, GRBF_Exponential
from .kernels import RBF_Matern, GRBF_Matern
from .kernels import RBF_Cubic, GRBF_Cubic
from .result import ResultDyCors
from .sampling import ERLatinHyperCube

from dask_jobqueue import SLURMCluster
from dask.distributed import Client

EPS = np.finfo(np.float64).eps

DEFAULT_OPTIONS = {"Nmax":250, "sig0":0.2, "sigm":0.2/2**6, "Ts":3, "Tf":5,
                   "weights":[0.3,0.5,0.8,0.95], "l":np.sqrt(0.5), "nu":5/2,
                   "optim_loo":False, "nits_loo":40, "warnings":True}

METHODS = ["RBF-Expo", "RBF-Matern", "RBF-Cubic",
           "GRBF-Expo", "GRBF-Matern", "GRBF-Cubic"]

NCORES = multiprocessing.cpu_count()
PAR_DEFAULT_OPTIONS = {"SLURM":False, "cores_per_feval":1,
                       "par_fevals":NCORES, "memory":"1GB",
                       "walltime":"00:10:00", "queue":"regular"}

def minimize(fun, x0=None, args=(), method="RBF-Cubic", jac=None, bounds=None, 
             options=None, restart=None, parallel=False, par_options=None,
             verbose=True):
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
    parallel : boolean, optional
        Whether or not to use parallel function evaluations. The 
        default is to run in serial.
    par_options : dict, optional
        A dictionary of options to set the task manager:

            SLURM : boolean
                Whether or not the computations are carried out by 
                SLURM. To use with supercomputers.
            cores_per_feval : int
                Number of cores to use in each function evaluation.
            par_fevals : int
                Number of function evaluations to run in parallel.
            memory : str
                Requested memory for each function evaluation. To use 
                only if SLURM is set to True.
            walltime : str
                Requested wall time for each function evaluation. To 
                use only if SLURM is set to True.
            queue : str
                Name of the partition. To use only if SLURM is set to 
                True.

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
        
    Notes
    -----
    The parallel function evaluations feature needs further testing.
    
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

    if parallel:
        if par_options is None:
            par_options = PAR_DEFAULT_OPTIONS
        else:
            for key in PAR_DEFAULT_OPTIONS:
                if key not in par_options.keys():
                    par_options[key] = PAR_DEFAULT_OPTIONS[key]
    else:
        par_options = None

    # ignore annoying warnings
    if not options["warnings"]:
        warnings.filterwarnings("ignore", category=sla.LinAlgWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    # run the optimization
    DyCorsMin = DyCorsMinimize(fun, x0, args, method, jac, bounds, options,
                               restart, parallel, par_options, verbose)
    return DyCorsMin()

class DyCorsMinimize:
    """Implementation of DyCors algorithm.
    
    For a full description of the different options see :func:`minimize`
    """
    def __init__(self, fun, x0, args, method, jac, bounds, options, restart,
                 parallel, par_options, verbose):
        self.fun = fun
        self.x0 = x0
        self.args = args
        self.method = method
        self.jac = jac
        self.bounds = bounds
        self.options = options
        self.restart = restart
        self.parallel = parallel
        self.par_options = par_options
        self.verbose = verbose
        
        self.l = self.options["l"]
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

        if self.method.startswith("G"):
            self.grad = True
        else:
            self.grad = False

        self.la = sla

        # Set parallel variables
        if self.parallel:
            self.SLURM = self.par_options["SLURM"]
            self.cores = self.par_options["cores_per_feval"]
            self.procs = self.par_options["par_fevals"]
            
            if self.SLURM:
                self.memory = self.par_options["memory"]
                self.wt = self.par_options["walltime"]
                self.queue = self.par_options["queue"]
        else:
            self.cores = 1
            self.procs = 1

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
            
        # compute starting points
        self.fevals = 0
        if self.restart is None:
            self.initialize()
        else:
            self.initialize_restart()
            
        self.n0, self.Np = self.m, self.Nmax - self.m
        
        self.k = min(100*self.d, 5000) # number of trial points
        self.Ts, self.Tf = self.options["Ts"], max(self.d, self.options["Tf"])
        self.weights = self.options["weights"]
        self.optim_loo = self.options["optim_loo"]
        self.nits_loo = self.options["nits_loo"]
            
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
        # First, set up Client
        if self.parallel and self.SLURM:
            cluster = SLURMCluster(n_workers=self.procs, cores=self.cores,
                                   processes=1, memory=self.memory, 
                                   walltime=self.wt, queue=self.queue)
            client = Client(cluster)
        elif self.parallel and not self.SLURM:
            client = Client(n_workers=self.procs, 
                            threads_per_worker=self.cores, processes=False)
        
        try:
            nits = 1

            if self.verbose:
                print("nits = %d"%(nits))
                
            while (self.fevals < self.Nmax):
                # update internal parameters
                if self.optim_loo and (self.ic+self.m)%self.nits_loo==0:
                    self.update_internal_params()
                    
                # fit response surface model
                if self.grad:
                    _ = self.kernel.fit(self.x, self.f, self.df)
                else:
                    _ = self.kernel.fit(self.x, self.f)

                # generate trial points and select the bests
                self.trial_points()
                self.select_new_pts()

                if self.parallel:
                    # submit parallel fevals
                    futf = client.map(self.par_fun, 
                                      [self.fun for k in range(self.procs)],
                                      self.xnew)
                    if self.grad:
                        futdf = client.map(self.par_fun, 
                                           [self.jac
                                            for k in range(self.procs)],
                                           self.xnew)
                    
                    # gather data
                    fnew = client.gather(futf)
                    self.fnew = np.asarray([k[0] for k in fnew])

                    if self.grad:
                        dfnew = client.gather(futdf)
                        self.dfnew = np.asarray(dfnew).flatten()
                else:
                    # run serial fevals
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

        if self.parallel:
            client.close()
        
        return ResultDyCors(fun=self.fB[0],
                            jac=self.dfB if self.grad else None,
                            nfev=self.fevals,
                            njev=self.fevals if self.grad else None,
                            nit=nits, status=warnflag, message=task_str,
                            x=np.asarray(self.xB), success=(warnflag==0),
                            m=self.m, hist=np.asarray(self.fBhist),
                            dhist=(np.asarray(self.dfBhist)
                                   if self.grad else None),
                            xres=self.x, fres=self.f,
                            gres=self.df if self.grad else None)

    def par_fun(self, fun, xnew):
        """Wrapper around the function to be evaluated. Needed in case
        of parallel function evaluations.
        
        Parameters
        ----------
        fun : callable
            Function to be evaluated.
        xnew : float
            Point where the function is to be evaluated.
        
        Returns
        -------
        f : float
            Value of the function `fun` at `xnew`.
        """
        f = np.apply_along_axis(fun, 1, [xnew], *self.args)
        
        return f

    def initialize(self):
        """Compute function and gradient evaluations of initial
        sampling points.
        """
        # counters
        self.ic, self.Cs, self.Cf = 0, 0, 0
        self.fevals += self.m

        # set up client
        if self.parallel and self.SLURM:
            cluster = SLURMCluster(n_workers=self.procs, cores=self.cores, 
                                   processes=1, memory=self.memory, 
                                   walltime=self.wt, queue=self.queue)
            client = Client(cluster)
        elif self.parallel and not self.SLURM:
            client = Client(n_workers=self.procs, 
                            threads_per_worker=self.cores,
                            processes=False)

        if self.bounds is not None:
            # enforce bounds
            bL = self.bL*np.ones_like(self.x0)
            bU = self.bU*np.ones_like(self.x0)
            self.x = np.clip(self.x0, bL, bU)
        else:
            self.x = self.x0.copy()

        # evaluate initial points
        if self.parallel:
            futf = client.map(self.par_fun, [self.fun for k in range(self.m)],
                              self.x)
            if self.grad:
                futdf = client.map(self.par_fun, 
                                   [self.jac for k in range(self.m)],
                                   self.x)

            # gather data
            f = client.gather(futf)
            self.f = np.asarray([k[0] for k in f] )

            if self.grad:
                df = client.gather(futdf)
                self.df = np.asarray(df).flatten()
        else:
            self.f = np.apply_along_axis(self.fun, 1, self.x, *self.args)
            if self.grad:
                self.df = np.apply_along_axis(self.jac, 1, self.x, 
                                              *self.args).flatten()
        
        if self.parallel:
            client.close()
            
    def initialize_restart(self):
        """Initialize optimization from a previous optimization.
        """
        self.x = self.restart["xres"]
        self.f = self.restart["fres"]
        if self.grad:
            self.df = self.restart["gres"]
        
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
        while len(self.xnew)<self.procs:
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
        for i in range(self.procs):
            if (self.fnew[i]<self.fB):
                self.xB, self.fB = self.xnew[i], [self.fnew[i]]
                if self.grad:
                    self.dfB = self.dfnew[i*self.d:(i+1)*self.d]
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

    def update_internal_params(self):
        """Optimize internal parameters based on the Leave-One-Out
        error using a differential evolution algorithm [1]_, [2]_. A
        constraint is applied to the condition number of the kernel
        matrix to ensure the smoothness of the function.
        
        Notes
        -----
        We are not using the gradient information at the moment to
        compute the LOO-Error.
        
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
        def constr_f(ip):
            if self.method.endswith("Expo"):
                l = ip
                
                kernel = RBF_Exponential(l)
                Phi,_ = kernel.fit(self.x, self.f)
                
                return nla.cond(Phi)
                
            elif self.method.endswith("Matern"):
                l  = ip[:-1]
                nu = ip[-1]
                
                kernel = RBF_Matern(l, nu)
                Phi,_ = kernel.fit(self.x, self.f)
                
                return nla.cond(Phi)
            elif self.method.endswith("Cubic"):
                l  = ip
                
                kernel = RBF_Cubic(l)
                Phi,_ = kernel.fit(self.x, self.f)
                
                return nla.cond(Phi)
            
        def error(ip):
            if self.method.endswith("Expo"):
                l = ip
                n = self.x.shape[0]
                
                kernel = RBF_Exponential(l)
                Phi,_ = kernel.fit(self.x, self.f)
                
                H_inv = sla.inv(Phi)
                H_inv2 = H_inv@H_inv

                return nla.norm(self.f.T@H_inv2@self.f/(n*np.diag(H_inv2)),
                                ord=1)

            elif self.method.endswith("Matern"):
                l  = ip[:-1]
                nu = ip[-1]
                n = self.x.shape[0]
                
                kernel = RBF_Matern(l, nu)
                Phi,_ = kernel.fit(self.x, self.f)
                
                H_inv = sla.inv(Phi)
                H_inv2 = H_inv@H_inv

                return nla.norm(self.f.T@H_inv2@self.f/(n*np.diag(H_inv2)),
                                ord=1)
            elif self.method.endswith("Cubic"):
                l  = ip
                n = self.x.shape[0]
                
                kernel = RBF_Cubic(l)
                Phi,_ = kernel.fit(self.x, self.f)
                
                H_inv = sla.inv(Phi)
                H_inv2 = H_inv@H_inv

                return nla.norm(self.f.T@H_inv2@self.f/(n*np.diag(H_inv2)),
                                ord=1)

        from scipy.optimize import NonlinearConstraint
        nlc = NonlinearConstraint(constr_f, 0, 0.1/EPS)
        if self.verbose:
            print("Updating internal params...")
        if self.method.endswith("Expo") or self.method.endswith("Cubic"):
            error0 = error(self.l)
            try:
                sol = differential_evolution(func=error,
                                             bounds=(((1e-5,1e1),)*self.d),
                                             constraints=(nlc),
                                             strategy="rand2bin", maxiter=100)
                if sol["fun"]<error0:
                    if self.verbose:
                        print("Updated")
                    self.l = sol["x"]
                    self.kernel.update(self.l)
                else:
                    if self.verbose:
                        print("Not updated")
            except np.linalg.LinAlgError:
                if self.verbose:
                    print("Not updated")
        elif self.method.endswith("Matern"):
            error0 = error(np.concatenate((self.l,[self.nu])))
            try:
                sol = differential_evolution(func=error, 
                                             bounds=(((1e-5,1e1),) * self.d 
                                                     + ((0.5,5e1),)),
                                             constraints=(nlc), 
                                             strategy="rand2bin", maxiter=100)
                if sol["fun"]<error0:
                    if self.verbose:
                        print("Updated")
                    self.l = sol["x"][:-1]
                    self.nu = sol["x"][-1]
                    self.kernel.update(self.l, self.nu)
                else:
                    if self.verbose:
                        print("Not updated")
            except np.linalg.LinAlgError:
                if self.verbose:
                    print("Not updated")
                
    def restart_dycors(self):
        """Restart DyCors keeping only the best point.
        """
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
        
