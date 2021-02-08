from math import log
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.special import gamma, kv
from scipy.optimize import OptimizeResult
import warnings

DEFAULT_OPTIONS = {"Nmax":50, "nrestart":6, "sigma":0.2, "Ts":3, "Tf":5, "solver":"scipy",\
        "l":np.sqrt(0.5), "nu":5/2, "warnings":True}
METHODS = ['RBF-Expo', 'RBF-Matern', 'GRBF']

def minimize(fun, x0, method='RBF-Expo', jac=None, bounds=None, tol=None, options=None, verbose=True):
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
    DyCorsMin = DyCorsMinimize(fun, x0, method, jac, bounds, tol, options, verbose)
    return DyCorsMin()

class DyCorsMinimize:
    def __init__(self, fun, x0, method, jac, bounds, tol, options, verbose):
        self.fun = fun
        self.x0 = x0
        self.method = method
        self.jac = jac
        self.bounds = bounds
        self.tol = tol
        self.options = options
        self.verbose = verbose

        if self.method=='RBF-Expo':
            self.surrogateFunc = self.surrogateRBF_Expo
            self.evalFunc      = self.evalRBF_Expo
        elif self.method=='RBF-Matern':
            self.surrogateFunc = self.surrogateRBF_Matern
            self.evalFunc      = self.evalRBF_Matern
        elif self.method=='GRBF':
            self.surrogateFunc = self.surrogateGRBF
            self.evalFunc      = self.evalGRBF

        if self.method=='GRBF':
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
        self.k           = min(100*self.d, 500) # numer of trial points
        if self.bounds is not None:
            self.sigm    = self.options["sigma"]*(self.bU - self.bL) # standard deviation for new point generation
        else:
            self.sigm    = self.options["sigma"]*np.ones((self.d,))
        self.sig         = self.sigm
        self.Ts, self.Tf = self.options["Ts"], max(self.d, self.options["Tf"])
        self.nrestart    = self.options["nrestart"] # number of restarts
        self.l           = self.options["l"] # internal parameter of the kernel
        self.nu          = self.options["nu"] # order of the Bessel function

        self.initialize() # compute starting points
        
        self.iB          = np.argmin(self.f) # find best solution
        self.xB, self.fB = self.x[self.iB,:], [self.f[self.iB]]
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
                    self.fnew = np.apply_along_axis(self.fun, 1, [self.xnew]) # f()
                    if self.grad:
                        self.dfnew = np.apply_along_axis(self.jac, 1, [self.xnew]).flatten() #df()
                    self.update()

                    self.ic += 1
                    if (self.ic%5 == 0 and self.verbose):
                        print('  fevals = %d | fmin = %.2e'%(self.Np*(kk)+self.ic+self.m, self.fB[0]))

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
                          jac=np.apply_along_axis(self.jac, 1, [self.xB]) if self.grad else None,
                          nfev=self.Np*(kk+1)+self.m,
                          njev=self.Np*(kk+1)+self.m if self.grad else None,
                          nit=kk+1, status=warnflag, message=task_str,
                          x=self.xB, success=(warnflag==0), hess_inv=None)

    def surrogateRBF_Expo(self):
        # build a surrogate surface using cubic RBF's + linear polynomial
        #   (self.x,self.f): high-dimensional points and associated costs f(x)
        n = self.x.shape[0]

        # RBF-matrix
        R   = -2*np.dot(self.x, self.x.T) + np.sum(self.x**2, axis=1)\
            + np.sum((self.x.T)**2, axis=0)[:,np.newaxis]
        Phi = np.exp(-R/(2*self.l**2))      # RBF-part
        P   = np.hstack((np.ones((n,1)), self.x)) # polynomial part
        Z   = np.zeros((self.d+1,self.d+1)) # zero matrix
        A   = np.block([[Phi,P],[P.T,Z]])   # patched together

        # right-hand side
        F     = np.zeros(n+self.d+1)
        F[:n] = self.f # rhs-vector
        # self.s = self.la.solve(A, F, assume_a='sym') # solution
        self.s = self.la.solve(A, F) # solution

    def evalRBF_Expo(self):
        # evaluate the surrogate surface at {y}
        #   self.x: RBF-points
        #   self.s: coefficient vector of surrogate surface
        #   self.yk: trial points
        y = np.array(self.yk)
        m = y.shape[0]

        # RBF-matrix (evaluated at {y})
        R   = -2*np.dot(self.x, y.T) + np.sum(y**2, axis=1)\
            + np.sum(self.x**2, axis=1)[:,np.newaxis]
        Phi = np.exp(-R.T/(2*self.l**2))    # RBF-part
        P   = np.hstack((np.ones((m,1)),y)) # polynomial part
        A   = np.block([Phi,P])             # patched together
        return np.dot(A, self.s)             # evaluation

    def surrogateRBF_Matern(self):
        # build a surrogate surface using cubic RBF's + linear polynomial
        #   (self.x,self.f): high-dimensional points and associated costs f(x)
        n = self.x.shape[0]

        # RBF-matrix
        R   = -2*np.dot(self.x, self.x.T) + np.sum(self.x**2, axis=1)\
            + np.sum((self.x.T)**2, axis=0)[:,np.newaxis]
        Phi = 2**(1-self.nu)/gamma(self.nu)*(np.sqrt(2*self.nu*R)/self.l)**self.nu\
            * kv(self.nu, np.sqrt(2*self.nu*R)/self.l) # RBF-part
        np.nan_to_num(Phi, copy=False, nan=1.0)        # NaN values (R=0) are 1.0 if you take the limit
        P   = np.hstack((np.ones((n,1)), self.x))      # polynomial part
        Z   = np.zeros((self.d+1,self.d+1))            # zero matrix
        A   = np.block([[Phi,P],[P.T,Z]])              # patched together

        # right-hand side
        F     = np.zeros(n+self.d+1)
        F[:n] = self.f # rhs-vector
        # self.s = self.la.solve(A, F, assume_a='sym') # solution
        self.s = self.la.solve(A, F) # solution

    def evalRBF_Matern(self):
        # evaluate the surrogate surface at {y}
        #   self.x: RBF-points
        #   self.s: coefficient vector of surrogate surface
        #   self.yk: trial points
        y = np.array(self.yk)
        m = y.shape[0]

        # RBF-matrix (evaluated at {y})
        R   = -2*np.dot(self.x, y.T) + np.sum(y**2, axis=1)\
            + np.sum(self.x**2, axis=1)[:,np.newaxis]
        Phi = 2**(1-self.nu)/gamma(self.nu)*(np.sqrt(2*self.nu*R.T)/self.l)**self.nu\
            * kv(self.nu, np.sqrt(2*self.nu*R.T)/self.l) # RBF-part
        np.nan_to_num(Phi, copy=False, nan=1.0)        # NaN values (R=0) are 1.0 if you take the limit
        P   = np.hstack((np.ones((m,1)),y))            # polynomial part
        A   = np.block([Phi,P])                        # patched together
        return np.dot(A, self.s)                        # evaluation

    def surrogateGRBF(self):
        # build a surrogate surface using cubic RBF's
        n = self.x.shape[0]

        # RBF-matrix
        R = -2*np.dot(self.x, self.x.T) + np.sum(self.x**2, axis=1)\
            + np.sum((self.x.T)**2, axis=0)[:,np.newaxis]
        Phi = np.exp(-R/(2*self.l**2))

        # right-hand side
        F = np.zeros(n*(self.d+1))

        # function value
        F[:n] = self.f

        # derivative function value
        F[n:n*(self.d+1)] = self.df 
        
        # creation of matrix Phi_d (first derivatives of the kernel) : potential error here 
        Phi_d = np.zeros((n,n*self.d))
        for i in range(n):
            for j in range(n):
                for k in range(self.d):
                    Phi_d[i,j*self.d+k] = -2*Phi[i,j]*(self.x[i,k]-self.x[j,k])/(2*self.l**2)
        
        #creation of matrix Phi_dd (second deruvatives of exponential kernel ) : potential error here
        Phi_dd = np.zeros((n*self.d,n*self.d))
        
        for i in range(n):
            for k in range(self.d):
                for j in range(n):
                    for l in range (self.d):
                        if i==j:
                            Phi_dd[i*self.d+k,j*self.d+l] = 0
                        else:
                            Phi_dd[i*self.d+k,j*self.d+l] = -2*Phi_d[i,j*self.d+l]*(self.x[i,k]-self.x[j,k])/(2*self.l**2)\
                                - (2*Phi[i,j]/(2*self.l**2) if k==l else 0)
                            
        A = np.block([[Phi,-Phi_d],[np.transpose(Phi_d),Phi_dd]])        
        
        # self.s = self.la.solve(A, F, assume_a='sym')  # solution
        self.s = self.la.solve(A, F)

    def evalGRBF(self): 
        y = np.array(self.yk)
        m = y.shape[0] # number of points
        n = self.x.shape[0] # number of sample points
        
        # RBF part of problem to solve
        R   = -2*np.dot(self.x, y.T) + np.sum(y**2, axis=1)\
            + np.sum(self.x**2, axis=1)[:,np.newaxis]
        Phi = np.exp(-R.T/(2*self.l**2))
        
        # New part of matrix d_Phi  (gradient info of kernel) : most likely place where error is 
        d_Phi = np.zeros((m,n*self.d))
        
        for i in range(m):
            for j in range(n):
                for k in range(self.d):
                    d_Phi[i,j*self.d+k] = -2*Phi[i,j]*(y[i,k]-self.x[j,k])/(2*self.l**2)

        A = np.block([[Phi,-d_Phi]])
        return np.dot(A,self.s)

    def initialize(self):
        if self.bounds is not None:
            bL = self.bL*np.ones_like(self.x0)
            bU = self.bU*np.ones_like(self.x0)
            self.x = np.clip(self.x0, bL, bU) # enforce bounds
        else:
            self.x = self.x0.copy()

        # evaluate initial points
        self.f = np.apply_along_axis(self.fun, 1, self.x) #f()
        if self.grad:
            self.df = np.apply_along_axis(self.jac, 1, self.x).flatten() #df()
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
        s = self.evalFunc() # estimate function value

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
        else:
            self.Cf += 1

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
            self.sigm = self.options["sigma"]*(self.bU - self.bL)
        else:
            self.sigm = self.options["sigma"]*np.ones((self.d,))
        self.sig = self.sigm
        self.ic, self.Cs, self.Cf = 0, 0, 0
