import numpy as np
import matplotlib.pyplot as plt

from DyCors import minimize, SLatinHyperCube, RLatinHyperCube

eps = np.finfo(np.float64).eps

def Rastrigin(x):
    f = 10*len(x) + sum(x*x - 10*np.cos(2*np.pi*x))
    return f

def df_Rastrigin(x):
    df = 2*x + 20*np.pi*np.sin(2*np.pi*x)
    return df

# parameters of the problem
m,d       = 12,2    # size of initial population, dimensionality
bounds    = np.array([[-2,2],]*d)  # bounds

x0 = np.outer(m*[1],bounds[:,0]) + np.outer(m*[1],bounds[:,1]-bounds[:,0])\
          *np.random.rand(m,d)

# x0 = np.outer(m*[1],bounds[:,0]) + np.outer(m*[1],bounds[:,1]-bounds[:,0])\
#           *SLatinHyperCube(m,d)

# x0 = np.outer(m*[1],bounds[:,0]) + np.outer(m*[1],bounds[:,1]-bounds[:,0])\
#           *RLatinHyperCube(m,d)

Nmax      = 50 # Maximum number of function evaluations per restart
nrestart  = 6 # number of restarts
sig0      = np.array([0.2]*d) # initial standard deviation for trial points
sigm      = np.array([1e3*eps]*d) # minimum standard deviation for trial points
Ts        = 3 
Tf        = 5
solver    = "scipy"
l         = 0.5*np.ones((d,))
nu        = 5/2
options   = {}
options   = {"Nmax":Nmax, "nrestart":nrestart, "sig0":sig0, "sigm":sigm, "Ts":Ts, "Tf":Tf,\
            "solver":solver, "l":l, "nu":nu, "optim_ip":False, "warnings":False}

solf = minimize(fun=Rastrigin, x0=x0, method='GRBF-Expo', jac=df_Rastrigin,\
    bounds=bounds, tol=None, options=options, verbose=True)

print(solf)

print('x_opt = ',solf["x"])
print('f(x_opt) = %.8f'%solf["fun"])
