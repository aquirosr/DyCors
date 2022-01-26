import numpy as np
import matplotlib.pyplot as plt

from DyCors import minimize, ERLatinHyperCube

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
          *ERLatinHyperCube(m,d)

Nmax     = 250
sig0     = np.array([0.2]*d)
sigm     = np.array([0.2/2**6]*d)
Ts       = 3
Tf       = 5
weights  = [0.3,0.5,0.8,0.95]
l        = 0.5*np.ones((d,))
nu       = 5/2
nits_loo = 40
options  = {"Nmax":Nmax, "sig0":sig0, "sigm":sigm, "Ts":Ts, "Tf":Tf, "l":l,
            "weights": weights, "nu":nu, "optim_loo":False,
            "nits_loo":nits_loo, "warnings":False}

# initial optimization
solf = minimize(fun=Rastrigin, x0=x0, args=(), method="RBF-Cubic",
                jac=df_Rastrigin, bounds=bounds, options=options,
                restart=None, verbose=True)

print(solf)

print("x_opt = ",solf["x"])
print("f(x_opt) = %.8f"%solf["fun"])

solf.plot()
plt.show()

# restarted optimization
Nmax = 500
options["Nmax"] = Nmax
solf = minimize(fun=Rastrigin, x0=None, args=(), method="RBF-Cubic",
                jac=df_Rastrigin, bounds=bounds, options=options,
                restart=solf, verbose=True)

print(solf)

print("x_opt = ",solf["x"])
print("f(x_opt) = %.8f"%solf["fun"])

solf.plot()
plt.show()