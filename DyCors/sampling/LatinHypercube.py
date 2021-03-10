import numpy as np

def LatinHyperCube(m,d):
    """Non-symmetric Latin Hypercube.
    
    Parameters
    ----------
    m : int
        Number of sampling points.
    d : int
        Number of dimensions.
    """

    delta = np.ones(d)/m
    X     = np.zeros((m,d))
    for j in range(d):
        for i in range(m):
            X[i,j] = (2*i+1)/2*delta[j]
            
    P = np.zeros((m,d),dtype=int)
    P[:,0] = np.arange(m)
    
    for j in range(1,d):
        P[:,j] = np.random.permutation(np.arange(m))
    
    IPts = np.zeros((m,d))
    for j in range(d):
        for i in range(m):
            IPts[i,j] = X[P[i,j],j]
    return IPts

if __name__ == "__main__":
    print('This is test for LHDstandard')
    dim = 2
    m = 5
    print('dim is ',dim)
    print('m is ',m)
    print('set seed to 5')
    np.random.seed(5)
    for i in range(3):
        print(LatinHyperCube(m,dim))
