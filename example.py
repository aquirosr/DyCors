import numpy as np
import matplotlib.pyplot as plt

from DyCors import *

eps = np.finfo(np.float64).eps

def Rastrigin(x):
    f = 10*len(x) + sum(x*x - 10*np.cos(2*np.pi*x))
    return f

def df_Rastrigin(x):
    df = 2*x + 20*np.pi*np.sin(2*np.pi*x)
    return df

# parameters of the problem
m,d       = 12,2
bounds    = np.array([[-2,2],]*d)

x0 = np.outer(m*[1],bounds[:,0]) + np.outer(m*[1],bounds[:,1]-bounds[:,0])\
          *np.random.rand(m,d)

# x0 = np.outer(m*[1],bounds[:,0]) + np.outer(m*[1],bounds[:,1]-bounds[:,0])\
#           *SLatinHyperCube(m,d)

# x0 = np.outer(m*[1],bounds[:,0]) + np.outer(m*[1],bounds[:,1]-bounds[:,0])\
#           *RLatinHyperCube(m,d)

Nmax     = 50
nrestart = 6
sig0     = np.array([0.2]*d)
sigm     = np.array([1e3*eps]*d)
Ts       = 3
Tf       = 5
l        = 0.5*np.ones((d,))
nu       = 5/2
options  = {"Nmax":Nmax, "nrestart":nrestart, "sig0":sig0, "sigm":sigm, "Ts":Ts, "Tf":Tf,\
            "l":l, "nu":nu, "optim_ip":False, "warnings":False}
parallel    = False
par_options = {'SLURM':False, 'cores_per_feval':1, 'par_fevals':4, 'memory':'1GB',\
                'walltime':'00:10:00', 'queue':'regular'}

solf = minimize(fun=Rastrigin, x0=x0, args=(), method='RBF-Expo', jac=df_Rastrigin, bounds=bounds,\
                tol=None, options=options, parallel=parallel, par_options=par_options, verbose=True)

print(solf)

print('x_opt = ',solf["x"])
print('f(x_opt) = %.8f'%solf["fun"])
