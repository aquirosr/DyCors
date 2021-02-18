from math import log
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.special import factorial
from scipy.optimize import OptimizeResult, differential_evolution
import warnings

EPS = np.finfo(np.float64).eps

DEFAULT_OPTIONS = {"Nmax":50, "nrestart":6, "sig0":0.2, "sigm":1e3*EPS, "Ts":3, "Tf":5,\
                    "solver":"scipy", "l":np.sqrt(0.5), "nu":5/2, "optim_ip":False, "warnings":True}
METHODS = ['RBF-Expo', 'RBF-Matern', 'GRBF-Expo', 'GRBF-Matern']

def minimize(fun, x0, args=(), method='RBF-Expo', jac=None, bounds=None, tol=None, options=None, verbose=True):
    # check options are ok
    if not callable(fun):
        raise TypeError('fun is not callable')

    if x0.ndim!=2:
        raise ValueError('dimensions of x0 are not 2')

    if method not in METHODS:
        raise ValueError('invalid method %s,\n  valid options are %s'%(method, METHODS))
    elif method=='GRBF':
        if not callable(jac):
            raise TypeError('jac is not callable')

    if bounds is not None and x0.shape[1]!=bounds.shape[0]:
        raise ValueError('dimension mismatch. Modify x0 or bounds')

    if options is None:
        options = DEFAULT_OPTIONS
    else:
        for key in DEFAULT_OPTIONS:
            if key not in options.keys():
                options[key] = DEFAULT_OPTIONS[key]
    
    solvers = ['numpy', 'scipy']
    if options["solver"] not in solvers:
        raise ValueError('invalid solver %s,\n  valid options are %s'%(options["solver"], solvers))

    # ignore annoying warnings
    if not options["warnings"]:
        warnings.filterwarnings("ignore", category=sla.LinAlgWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    # run the optimization
    DyCorsMin = DyCorsMinimize(fun, x0, args, method, jac, bounds, tol, options, verbose)
    return DyCorsMin()

class DyCorsMinimize:
    def __init__(self, fun, x0, args, method, jac, bounds, tol, options, verbose):
        self.fun = fun
        self.x0 = x0
        self.args = args
        self.method = method
        self.jac = jac
        self.bounds = bounds
        self.tol = tol
        self.options = options
        self.verbose = verbose

        if self.method=='RBF-Expo':
            self.surrogateFunc = self.surrogateRBF_Expo
            self.evalSurr      = self.evalRBF_Expo
        elif self.method=='RBF-Matern':
            self.surrogateFunc = self.surrogateRBF_Matern
            self.evalSurr      = self.evalRBF_Matern
        elif self.method=='GRBF-Expo':
            self.surrogateFunc = self.surrogateGRBF_Expo
            self.evalSurr      = self.evalGRBF_Expo
        elif self.method=='GRBF-Matern':
            self.surrogateFunc = self.surrogateGRBF_Matern
            self.evalSurr      = self.evalGRBF_Matern

        if self.method=='GRBF-Expo' or self.method=='GRBF-Matern':
            self.grad = True
        else:
            self.grad = False

        if options["solver"]=='numpy':
            self.la = nla
        elif options["solver"]=='scipy':
            self.la = sla

        self.m, self.d   = self.x0.shape # size of initial population, dimensionality
        if self.bounds is not None:
            self.bL, self.bU = self.bounds[:,0], self.bounds[:,1] # bounds
        self.Nmax        = self.options["Nmax"] # maximum number of function evaluations per restart
        self.n0, self.Np = self.m, self.Nmax - self.m
        self.ic          = 0 # counter
        self.k           = min(100*self.d, 500) # number of trial points
        if self.bounds is not None:
            self.sigm    = self.options["sigm"]*np.ones((self.d,)) # minimum standard deviation
            self.sig     = self.options["sig0"]*(self.bU - self.bL) # initial standard deviation
        else:
            self.sigm    = self.options["sigm"]*np.ones((self.d,)) # minimum standard deviation
            self.sig     = self.options["sig0"]*np.ones((self.d,)) # initial standard deviation
        self.Ts, self.Tf = self.options["Ts"], max(self.d, self.options["Tf"])
        self.nrestart    = self.options["nrestart"] # number of restarts
        self.l           = self.options["l"]*np.ones((self.d,)) # starting kernel width
        self.nu          = self.options["nu"] # starting order of the Bessel function
        self.optim_ip    = self.options["optim_ip"] # Optimize internal parameters?

        self.initialize() # compute starting points
        self.fevals = self.m # function evaluations
        
        self.iB          = np.argmin(self.f) # find best solution
        self.xB, self.fB = self.x[self.iB,:], np.asarray([self.f[self.iB]])
        self.fBs         = np.zeros((self.nrestart+1,), dtype=np.float64)
        self.fBs[0]      = self.fB[0]
        self.converged   = False

    def __call__(self):
        try:
            for kk in range(self.nrestart):
                if self.verbose:
                    print('nits = %d'%(kk+1))
                while (self.ic < self.Np):
                    self.surrogateFunc() # fit response surface model

                    self.trialPoints() # trial points
                    self.selectNewPt()
                    self.fnew = np.apply_along_axis(self.fun, 1, [self.xnew], *self.args) # f()
                    if self.grad:
                        self.dfnew = np.apply_along_axis(self.jac, 1, [self.xnew], *self.args).flatten() #df()
                    self.update()

                    self.ic += 1
                    self.fevals = self.Np*(kk)+self.ic+self.m
                    if (self.fevals%5 == 0 and self.verbose):
                        print('  fevals = %d | fmin = %.2e'%(self.fevals, self.fB[0]))

                self.fBs[kk+1] = self.fB[0]
                if self.tol is not None:
                    self.checkConvergence(self.fBs[:kk+2])
                if self.converged:
                    task_str = 'CONVERGED'
                    warnflag = 0
                    break

                self.restart() # restart optimization

            if not self.converged:
                task_str = 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
                if self.tol is None:
                    warnflag = 0
                else:
                    warnflag = 1

        except np.linalg.LinAlgError:
            task_str = 'STOP: SINGULAR MATRIX'
            warnflag = 2

        return OptimizeResult(fun=self.fB,
                          jac=np.apply_along_axis(self.jac, 1, [self.xB], *self.args) if self.grad else None,
                          nfev=self.fevals,
                          njev=self.fevals if self.grad else None,
                          nit=kk+1, status=warnflag, message=task_str,
                          x=self.xB, success=(warnflag==0), hess_inv=None)

    def surrogateRBF_Expo(self):
        # build a surrogate surface using cubic RBF's + linear polynomial
        #   (self.x,self.f): high-dimensional points and associated costs f(x)
        n = self.x.shape[0]

        # RBF-matrix
        R   = -2*np.dot(self.x/self.l, self.x.T/self.l[:,np.newaxis]) + np.sum(self.x**2/self.l**2, axis=1)\
            + np.sum(self.x.T**2/self.l[:,np.newaxis]**2, axis=0)[:,np.newaxis]
        Phi = np.exp(-R/2)      # RBF-part

        P = np.hstack((np.ones((n,1)), self.x)) # polynomial part
        Z = np.zeros((self.d+1,self.d+1)) # zero matrix
        A = np.block([[Phi,P],[P.T,Z]])   # patched together

        # right-hand side
        F     = np.zeros(n+self.d+1)
        F[:n] = self.f # rhs-vector
        # self.s = self.la.solve(A, F, assume_a='sym') # solution
        self.s = self.la.solve(A, F) # solution

    def evalRBF_Expo(self, yk):
        # evaluate the surrogate surface at {y}
        #   self.x: RBF-points
        #   self.s: coefficient vector of surrogate surface
        #   self.yk: trial points
        y = np.array(yk)
        m = y.shape[0]

        # RBF-matrix (evaluated at {y})
        R   = -2*np.dot(self.x/self.l, y.T/self.l[:,np.newaxis]) + np.sum(y**2/self.l**2, axis=1)\
            + np.sum(self.x**2/self.l**2, axis=1)[:,np.newaxis]
        Phi = np.exp(-R.T/2)    # RBF-part

        P = np.hstack((np.ones((m,1)),y)) # polynomial part
        A = np.block([Phi,P])             # patched together
        return np.dot(A, self.s)             # evaluation

    def surrogateRBF_Matern(self):
        # Half integer simplification of Matérn Kernel
        #   (self.x,self.f): high-dimensional points and associated costs f(x)
        p = int(round(self.nu-1/2)+1e-8)
        n = self.x.shape[0]

        # RBF-matrix
        R = -2*np.dot(self.x/self.l, self.x.T/self.l[:,np.newaxis]) + np.sum(self.x**2/self.l**2, axis=1)\
            + np.sum(self.x.T**2/self.l[:,np.newaxis]**2, axis=0)[:,np.newaxis]
        R[R<=0.0] = EPS

        Phi = factorial(p)/factorial(2*p)*np.exp(-np.sqrt((2*p+1)*R))
        tmp = np.zeros_like(Phi)
        for i in range(p+1):
            tmp += factorial(p+i)/(factorial(i)*factorial(p-i))\
                * (2*np.sqrt((2*p+1)*R))**(p-i)
        Phi *= tmp

        P = np.hstack((np.ones((n,1)), self.x))      # polynomial part
        Z = np.zeros((self.d+1,self.d+1))            # zero matrix
        A = np.block([[Phi,P],[P.T,Z]])              # patched together

        # right-hand side
        F     = np.zeros(n+self.d+1)
        F[:n] = self.f # rhs-vector
        # self.s = self.la.solve(A, F, assume_a='sym') # solution
        self.s = self.la.solve(A, F) # solution

    def evalRBF_Matern(self, yk):
        # Half integer simplification of Matérn Kernel
        # evaluate the surrogate surface at {y}
        #   self.x: RBF-points
        #   self.s: coefficient vector of surrogate surface
        #   self.yk: trial points
        p = int(round(self.nu-1/2)+1e-8)

        y = np.array(yk)
        m = y.shape[0]

        # RBF-matrix (evaluated at {y})
        R = -2*np.dot(self.x/self.l, y.T/self.l[:,np.newaxis]) + np.sum(y**2/self.l**2, axis=1)\
            + np.sum(self.x**2/self.l**2, axis=1)[:,np.newaxis]
        R[R<=0.0] = EPS

        Phi = factorial(p)/factorial(2*p)*np.exp(-np.sqrt((2*p+1)*R.T))
        tmp = np.zeros_like(Phi)
        for i in range(p+1):
            tmp += factorial(p+i)/(factorial(i)*factorial(p-i))\
                * (2*np.sqrt((2*p+1)*R.T))**(p-i)
        Phi *= tmp

        P = np.hstack((np.ones((m,1)),y))            # polynomial part
        A = np.block([Phi,P])                        # patched together
        return np.dot(A, self.s)                        # evaluation

    def surrogateGRBF_Expo(self):
        n = self.x.shape[0]

        # RBF-matrix
        R   = -2*np.dot(self.x/self.l, self.x.T/self.l[:,np.newaxis]) + np.sum(self.x**2/self.l**2, axis=1)\
            + np.sum(self.x.T**2/self.l[:,np.newaxis]**2, axis=0)[:,np.newaxis]
        Phi = np.exp(-R/2) 
        
        # First derivative
        _Phi_d = np.zeros((n,n,self.d))
        _Phi_d = -2*Phi[...,np.newaxis] * (self.x[:,np.newaxis,:] - self.x[np.newaxis,:,:])\
            / (2*self.l[np.newaxis,np.newaxis,:]**2)
        Phi_d = _Phi_d.reshape((n,n*self.d))

        # Second derivative
        Phi_dd = np.zeros((n,self.d,n,self.d))
        Phi_dd = - 2*_Phi_d[:,np.newaxis,:,:]\
            * (self.x[:,:,np.newaxis,np.newaxis] - self.x.T[np.newaxis,:,:,np.newaxis])\
            / (2*self.l[np.newaxis,:,np.newaxis,np.newaxis]**2)\
            - np.diag(np.ones(self.d))[np.newaxis,:,np.newaxis,:]\
            * 2*Phi[:,np.newaxis,:,np.newaxis] / (2*self.l[np.newaxis,:,np.newaxis,np.newaxis]**2)
        Phi_dd = Phi_dd.reshape((n*self.d,n*self.d))

        A = np.block([[Phi,Phi_d],[-np.transpose(Phi_d),Phi_dd]])

        # right-hand side
        F = np.zeros(n*(self.d+1))

        # function value
        F[:n] = self.f

        # derivative function value
        F[n:n*(self.d+1)] = self.df
        
        # self.s = self.la.solve(A, F, assume_a='sym')  # solution
        self.s = self.la.solve(A, F)

    def evalGRBF_Expo(self, yk): 
        y = np.array(yk)
        m = y.shape[0] # number of points
        n = self.x.shape[0] # number of sample points
        
        # RBF part of problem to solve
        R   = -2*np.dot(self.x/self.l, y.T/self.l[:,np.newaxis]) + np.sum(y**2/self.l**2, axis=1)\
            + np.sum(self.x**2/self.l**2, axis=1)[:,np.newaxis]
        Phi = np.exp(-R.T/2)
        
        # First derivative 
        d_Phi = np.zeros((m,n,self.d))
        d_Phi = -2*Phi[...,np.newaxis] * (y[:,np.newaxis,:] - self.x[np.newaxis,:,:])\
            / (2*self.l[np.newaxis,np.newaxis,:]**2)
        d_Phi = d_Phi.reshape((m,n*self.d))

        A = np.block([[Phi,d_Phi]])
        return np.dot(A,self.s)

    def surrogateGRBF_Matern(self):
        # Half integer simplification of Matérn Kernel
        p = int(round(self.nu-1/2)+1e-8)

        n = self.x.shape[0]

        # RBF-matrix
        R = -2*np.dot(self.x/self.l, self.x.T/self.l[:,np.newaxis]) + np.sum(self.x**2/self.l**2, axis=1)\
            + np.sum(self.x.T**2/self.l[:,np.newaxis]**2, axis=0)[:,np.newaxis]
        R[R<=0.0] = EPS # R=0.0 is indeterminate
        
        # temporary matrices
        tmp0 = np.zeros_like(R)
        tmp1 = np.zeros_like(R)
        tmp2 = np.zeros_like(R)
        for i in range(p+1):
            tmp0 += factorial(p+i) / (factorial(i) * factorial(p-i))\
                * (2*np.sqrt((2*p+1)*R))**(p-i)
            if i<p:
                tmp1 += factorial(p+i) / (factorial(i) * factorial(p-i))\
                   * (p-i) * (2*np.sqrt((2*p+1)*R))**(p-i-1) * 2*np.sqrt(2*p+1)
            if i<p-1:
                tmp2 += factorial(p+i) / (factorial(i) * factorial(p-i))\
                    * (p-i) * (p-i-1) * (2*np.sqrt((2*p+1)*R))**(p-i-2) * (2*np.sqrt(2*p+1))**2

        fp_f2p_er = factorial(p) / factorial(2*p) * np.exp(-np.sqrt((2*p+1)*R))

        Phi = fp_f2p_er * tmp0

        # First derivative
        Phi_d = np.zeros((n,n,self.d))
        Phi_d = (Phi[:,:,np.newaxis] * (-np.sqrt(2*p+1))\
                + fp_f2p_er[:,:,np.newaxis] * tmp1[:,:,np.newaxis])\
            * (self.x[:,np.newaxis,:] - self.x[np.newaxis,:,:])\
            / np.sqrt(R[:,:,np.newaxis]) / self.l[np.newaxis,np.newaxis,:]**2
        Phi_d = Phi_d.reshape((n,n*self.d))

        # Second derivative
        Phi_dd = np.zeros((n,self.d,n,self.d))
        Phi_dd = (Phi[:,np.newaxis,:,np.newaxis] * (-np.sqrt(2*p+1))**2\
                + 2 * fp_f2p_er[:,np.newaxis,:,np.newaxis] * (-np.sqrt(2*p+1)) * tmp1[:,np.newaxis,:,np.newaxis]\
                + fp_f2p_er[:,np.newaxis,:,np.newaxis] * tmp2[:,np.newaxis,:,np.newaxis])\
            * (self.x[:,np.newaxis,np.newaxis,:] - self.x[np.newaxis,np.newaxis,:,:])\
            / np.sqrt(R[:,np.newaxis,:,np.newaxis]) / self.l[np.newaxis,np.newaxis,np.newaxis,:]**2\
            * (self.x[:,:,np.newaxis,np.newaxis] - self.x.T[np.newaxis,:,:,np.newaxis])\
            / np.sqrt(R[:,np.newaxis,:,np.newaxis]) / self.l[np.newaxis,:,np.newaxis,np.newaxis]**2\
            + (Phi[:,np.newaxis,:,np.newaxis] * (-np.sqrt(2*p+1))\
                + fp_f2p_er[:,np.newaxis,:,np.newaxis] * tmp1[:,np.newaxis,:,np.newaxis])\
            * (np.diag(np.ones(self.d))[np.newaxis,:,np.newaxis,:]\
                * (np.sqrt(R[:,np.newaxis,:,np.newaxis] * self.l[np.newaxis,:,np.newaxis,np.newaxis]**4))\
                - (self.x[:,:,np.newaxis,np.newaxis] - self.x.T[np.newaxis,:,:,np.newaxis])\
                * (self.x[:,np.newaxis,np.newaxis,:] - self.x[np.newaxis,np.newaxis,:,:])\
                / np.sqrt(R[:,np.newaxis,:,np.newaxis]) / self.l[np.newaxis,np.newaxis,np.newaxis,:]**2\
                * self.l[np.newaxis,:,np.newaxis,np.newaxis]**2)\
            / (np.sqrt(R[:,np.newaxis,:,np.newaxis])\
                * self.l[np.newaxis,:,np.newaxis,np.newaxis]**2)**2
        Phi_dd = Phi_dd.reshape((n*self.d,n*self.d))

        A = np.block([[Phi,Phi_d],[-np.transpose(Phi_d),Phi_dd]])

        # right-hand side
        F = np.zeros(n*(self.d+1))

        # function value
        F[:n] = self.f

        # derivative function value
        F[n:n*(self.d+1)] = self.df 

        # self.s = self.la.solve(A, F, assume_a='sym')  # solution
        self.s = self.la.solve(A, F)

    def evalGRBF_Matern(self, yk):
        # Half integer simplification of Matérn Kernel
        p = int(round(self.nu-1/2)+1e-8)

        y = np.array(yk)
        m = y.shape[0] # number of points
        n = self.x.shape[0] # number of sample points
        
        # RBF part of problem to solve
        R   = -2*np.dot(self.x/self.l, y.T/self.l[:,np.newaxis]) + np.sum(y**2/self.l**2, axis=1)\
            + np.sum(self.x**2/self.l**2, axis=1)[:,np.newaxis]
        R[R<=0.0] = EPS # R=0.0 is indeterminate

        # temporary matrices
        tmp0 = np.zeros_like(R.T)
        tmp1 = np.zeros_like(R.T)
        for i in range(p+1):
            tmp0 += factorial(p+i)/(factorial(i)*factorial(p-i))\
                * (2*np.sqrt((2*p+1)*R.T))**(p-i)
            if i<p:
                tmp1 += factorial(p+i) / (factorial(i) * factorial(p-i))\
                    * (p-i) * (2*np.sqrt((2*p+1)*R.T))**(p-i-1) * 2*np.sqrt(2*p+1)

        fp_f2p_er = factorial(p) / factorial(2*p) * np.exp(-np.sqrt((2*p+1)*R.T))

        Phi = fp_f2p_er * tmp0

        # First derivative             
        d_Phi = np.zeros((m,n,self.d))
        d_Phi = (Phi[:,:,np.newaxis] * (-np.sqrt(2*p+1))\
            + fp_f2p_er[:,:,np.newaxis] * tmp1[:,:,np.newaxis])\
            * (y[:,np.newaxis,:]-self.x[np.newaxis,:,:])\
            / np.sqrt(R.T[:,:,np.newaxis]) / self.l[np.newaxis,np.newaxis,:]**2
        d_Phi = d_Phi.reshape((m,n*self.d))

        A = np.block([[Phi,d_Phi]])
        return np.dot(A,self.s)

    def initialize(self):
        if self.bounds is not None:
            bL = self.bL*np.ones_like(self.x0)
            bU = self.bU*np.ones_like(self.x0)
            self.x = np.clip(self.x0, bL, bU) # enforce bounds
        else:
            self.x = self.x0.copy()

        # evaluate initial points
        self.f = np.apply_along_axis(self.fun, 1, self.x, *self.args) #f()
        if self.grad:
            self.df = np.apply_along_axis(self.jac, 1, self.x, *self.args).flatten() #df()
        self.Cs, self.Cf = 0, 0     # counter: success, failure

    def trialPoints(self):
        p  = min(20/self.d, 1)*(1-log(self.ic+1)/log(self.Np)) # probability for coordinates
        om = np.random.rand(self.d) # select coordinates to perturb
        Ip = np.argwhere(om<=p).flatten()
        if (len(Ip)==0): Ip = np.random.randint(0, self.d)
        M  = np.zeros((self.k,self.d),dtype=int) # mask
        M[:,Ip] = 1
        yk = np.outer(self.k*[1], self.xB) + M*np.random.normal(0, self.sig, (self.k,self.d)) # generate trial points
        if self.bounds is not None:
            bL = self.bL*np.ones_like(yk)
            bU = self.bU*np.ones_like(yk)
            self.yk = np.clip(yk,bL,bU) # enforce bounds
        else:
            self.yk = yk.copy()

    def selectNewPt(self):
        n = self.x.shape[0]
        s = self.evalSurr(self.yk) # estimate function value

        # compute RBF-score
        s1, s2 = s.min(), s.max()
        VR     = (s-s1)/(s2-s1) if (abs(s2-s1)>1.e-8) else 1

        # compute DIS-score
        dis = np.zeros(self.k)
        for i in range(self.k):
            tmp1   = np.outer(n*[1], self.yk[i,:]) - self.x
            tmp2   = np.apply_along_axis(nla.norm, 0, tmp1)
            dis[i] = tmp2.min()
        d1, d2 = dis.min(), dis.max()
        VD     = (d2-dis)/(d2-d1) if (abs(d2-d1)>1.e-8) else 1

        # combine the two scores
        G      = [0.3,0.5,0.8,0.95] # weight rotation
        nn     = (self.ic+1)%4
        nnn    = nn - 1 if (nn!=0) else 3
        wR, wD = G[nnn], 1 - G[nnn]
        V      = wR*VR + wD*VD # full weight

        # select the next point (avoid singular RBF matrix)
        iB, iP = np.argsort(V), 0
        xnew  = self.yk[iB[iP],:]
        while (xnew.tolist() in self.x.tolist()):
            iP  +=1
            if iP>=iB.shape[0]:
                break
            xnew = self.yk[iB[iP],:]
        self.xnew = xnew

    def checkConvergence(self, fBs):
        if np.abs(fBs[-1] - fBs[-2])/fBs[-2]<=self.tol:
            self.converged = True

    def update(self):
        # update counters
        if (self.fnew<self.fB):
            self.xB, self.fB = self.xnew, self.fnew
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
        self.x  = np.vstack((self.x, self.xnew[np.newaxis,:]))
        self.f  = np.concatenate((self.f, self.fnew))
        if self.grad:
            self.df  = np.concatenate((self.df, self.dfnew))
        
        # update internal parameters
        if self.optim_ip and self.fevals%10==0 and self.ic>20:
            self.update_internal_params()

    def update_internal_params(self):
        if self.method=='RBF-Expo' or self.method=='GRBF-Expo':
            def error(l):
                n = self.x.shape[0]

                R   = -2*np.dot(self.x/l, self.x.T/l[:,np.newaxis]) + np.sum(self.x**2/l**2, axis=1)\
                    + np.sum(self.x.T**2/l[:,np.newaxis]**2, axis=0)[:,np.newaxis]
                Phi = np.exp(-R/2)

                if self.method=='RBF-Expo':
                    H_inv = sla.inv(Phi)
                    H_inv2 = H_inv@H_inv

                    return nla.norm(self.f.T@H_inv2@self.f/(n*np.diag(H_inv2)), ord=1)

                elif self.method=='GRBF-Expo':
                    # First derivative
                    _Phi_d = np.zeros((n,n,self.d))
                    _Phi_d = -2*Phi[...,np.newaxis] * (self.x[:,np.newaxis,:] - self.x[np.newaxis,:,:])\
                        / (2*l[np.newaxis,np.newaxis,:]**2)
                    Phi_d = _Phi_d.reshape((n,n*self.d))

                    # Second derivative
                    Phi_dd = np.zeros((n,self.d,n,self.d))
                    Phi_dd = - 2*_Phi_d[:,np.newaxis,:,:]\
                        * (self.x[:,:,np.newaxis,np.newaxis] - self.x.T[np.newaxis,:,:,np.newaxis])\
                        / (2*l[np.newaxis,:,np.newaxis,np.newaxis]**2)\
                        - np.diag(np.ones(self.d))[np.newaxis,:,np.newaxis,:]\
                        * 2*Phi[:,np.newaxis,:,np.newaxis] / (2*l[np.newaxis,:,np.newaxis,np.newaxis]**2)
                    Phi_dd = Phi_dd.reshape((n*self.d,n*self.d))

                    A = np.block([[Phi,Phi_d],[-np.transpose(Phi_d),Phi_dd]])

                    H_inv = sla.inv(A)
                    H_inv2 = H_inv@H_inv

                    f = np.zeros(n*(self.d+1))
                    f[:n] = self.f
                    f[n:n*(self.d+1)] = self.df 

                    return nla.norm(f.T@H_inv2@f/(n*(self.d+1)*np.diag(H_inv2)), ord=1)

        elif self.method=='RBF-Matern' or self.method=='GRBF-Matern':
            def error(lnu):
                l  = lnu[:-1]
                nu = lnu[-1]
                p = int(round(nu-1/2)+1e-8)
                
                n = self.x.shape[0]

                R   = -2*np.dot(self.x/l, self.x.T/l[:,np.newaxis]) + np.sum(self.x**2/l**2, axis=1)\
                    + np.sum(self.x.T**2/l[:,np.newaxis]**2, axis=0)[:,np.newaxis]
                R[R<=0.0] = EPS # R=0.0 is indeterminate
                
                Phi = factorial(p) / factorial(2*p) * np.exp(-np.sqrt((2*p+1)*R))
                tmp = np.zeros_like(Phi)
                for i in range(p+1):
                    tmp += factorial(p+i) / (factorial(i) * factorial(p-i))\
                        * (2*np.sqrt((2*p+1)*R))**(p-i)
                Phi *= tmp

                if self.method=='RBF-Matern':
                    H_inv = sla.inv(Phi)
                    H_inv2 = H_inv@H_inv

                    return nla.norm(self.f.T@H_inv2@self.f/(n*np.diag(H_inv2)), ord=1)

                elif self.method=='GRBF-Matern':
                    # temporary matrices
                    tmp1 = np.zeros_like(R)
                    tmp2 = np.zeros_like(R)
                    for i in range(p):
                        tmp1 += factorial(p+i) / (factorial(i) * factorial(p-i))\
                            * (p-i) * (2*np.sqrt((2*p+1)*R))**(p-i-1) * 2*np.sqrt(2*p+1)
                        if i<p-1:
                            tmp2 += factorial(p+i) / (factorial(i) * factorial(p-i))\
                                * (p-i) * (p-i-1) * (2*np.sqrt((2*p+1)*R))**(p-i-2) * (2*np.sqrt(2*p+1))**2

                    fp_f2p_er = factorial(p) / factorial(2*p) * np.exp(-np.sqrt((2*p+1)*R))

                    # First derivative
                    Phi_d = np.zeros((n,n,self.d))
                    Phi_d = (Phi[:,:,np.newaxis] * (-np.sqrt(2*p+1))\
                            + fp_f2p_er[:,:,np.newaxis] * tmp1[:,:,np.newaxis])\
                        * (self.x[:,np.newaxis,:] - self.x[np.newaxis,:,:])\
                        / np.sqrt(R[:,:,np.newaxis]) / l[np.newaxis,np.newaxis,:]**2
                    Phi_d = Phi_d.reshape((n,n*self.d))

                    # Second derivative
                    Phi_dd = np.zeros((n,self.d,n,self.d))
                    Phi_dd = (Phi[:,np.newaxis,:,np.newaxis] * (-np.sqrt(2*p+1))**2\
                            + 2 * fp_f2p_er[:,np.newaxis,:,np.newaxis] * (-np.sqrt(2*p+1)) * tmp1[:,np.newaxis,:,np.newaxis]\
                            + fp_f2p_er[:,np.newaxis,:,np.newaxis] * tmp2[:,np.newaxis,:,np.newaxis])\
                        * (self.x[:,np.newaxis,np.newaxis,:] - self.x[np.newaxis,np.newaxis,:,:])\
                        / np.sqrt(R[:,np.newaxis,:,np.newaxis]) / l[np.newaxis,np.newaxis,np.newaxis,:]**2\
                        * (self.x[:,:,np.newaxis,np.newaxis] - self.x.T[np.newaxis,:,:,np.newaxis])\
                        / np.sqrt(R[:,np.newaxis,:,np.newaxis]) / l[np.newaxis,:,np.newaxis,np.newaxis]**2\
                        + (Phi[:,np.newaxis,:,np.newaxis] * (-np.sqrt(2*p+1))\
                            + fp_f2p_er[:,np.newaxis,:,np.newaxis] * tmp1[:,np.newaxis,:,np.newaxis])\
                        * (np.diag(np.ones(self.d))[np.newaxis,:,np.newaxis,:]\
                            * (np.sqrt(R[:,np.newaxis,:,np.newaxis] * l[np.newaxis,:,np.newaxis,np.newaxis]**4))\
                            - (self.x[:,:,np.newaxis,np.newaxis] - self.x.T[np.newaxis,:,:,np.newaxis])\
                            * (self.x[:,np.newaxis,np.newaxis,:] - self.x[np.newaxis,np.newaxis,:,:])\
                            / np.sqrt(R[:,np.newaxis,:,np.newaxis]) / l[np.newaxis,np.newaxis,np.newaxis,:]**2\
                            * l[np.newaxis,:,np.newaxis,np.newaxis]**2)\
                        / (np.sqrt(R[:,np.newaxis,:,np.newaxis])\
                            * l[np.newaxis,:,np.newaxis,np.newaxis]**2)**2
                    Phi_dd = Phi_dd.reshape((n*self.d,n*self.d))                    
                    
                    A = np.block([[Phi,Phi_d],[-np.transpose(Phi_d),Phi_dd]])

                    H_inv = sla.inv(A)
                    H_inv2 = H_inv@H_inv

                    f = np.zeros(n*(self.d+1))
                    f[:n] = self.f
                    f[n:n*(self.d+1)] = self.df 

                    return nla.norm(f.T@H_inv2@f/(n*(self.d+1)*np.diag(H_inv2)), ord=1)

        if self.method=='RBF-Expo' or self.method=='GRBF-Expo':
            error0 = error(self.l)
            try:
                sol = differential_evolution(func=error, bounds=(((1e-2,2e0),)*self.d), strategy='rand2bin', maxiter=100)
                if sol["fun"]<error0:
                    self.l = sol["x"]
            except np.linalg.LinAlgError:
                pass
        elif self.method=='RBF-Matern' or self.method=='GRBF-Matern':
            error0 = error(np.concatenate((self.l,[self.nu])))
            try:
                sol = differential_evolution(func=error, bounds=(((1e-2,2e0),)*self.d+((0.5,2e1),)), strategy='rand2bin', maxiter=100)
                if sol["fun"]<error0:
                    self.l = sol["x"][:-1]
                    self.nu = sol["x"][-1]
            except np.linalg.LinAlgError:
                pass

    def restart(self):
        n    = self.x.shape[0]
        dis  = np.zeros(n)
        tmp1 = np.outer(n*[1], self.xB) - self.x
        dis  = np.apply_along_axis(nla.norm,1,tmp1)
        iS   = np.argsort(dis)

        self.x, self.f = self.x[iS[:self.n0],:], self.f[iS[:self.n0]]
        if self.grad:
            df = np.zeros((self.n0*self.d,))
            for k,iSk in enumerate(iS[:self.n0]):
                df[k*self.d:(k+1)*self.d] = self.df[iSk*self.d:(iSk+1)*self.d]
            self.df = df.copy()

        # reset the bounds and counters
        if self.bounds is not None:
            self.sigm = self.options["sigm"]*np.ones((self.d,))
            self.sig = self.options["sig0"]*(self.bU - self.bL)
        else:
            self.sigm = self.options["sigm"]*np.ones((self.d,))
            self.sig = self.options["sig0"]*np.ones((self.d,))
        self.ic, self.Cs, self.Cf = 0, 0, 0
