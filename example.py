import numpy as np
import matplotlib.pyplot as plt

from src import minimize

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
          *np.random.rand(m,d) # random

Nmax      = 50 # Maximum number of function evaluations per restart
nrestart  = 6 # number of restarts
sigma     = np.array([0.2]*d) # initial standard deviation for trial points
Ts        = 3 
Tf        = 5
solver    = "scipy"
l         = 0.5
nu        = 5/2
options   = {}
options   = {"Nmax":Nmax, "nrestart":nrestart, "sigma":sigma, "Ts":Ts, "Tf":Tf,\
    "solver":solver, "l":l, "nu":nu, "warnings":False}

solf = minimize(fun=Rastrigin, x0=x0, method='GRBF', jac=df_Rastrigin,\
    bounds=bounds, tol=None, options=options, verbose=True)

print(solf)

print('x_opt = ',solf["x"])
print('f(x_opt) = %.8f'%solf["fun"])
