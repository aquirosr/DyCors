import numpy as np

def RLatinHyperCube(m,d):
    """Non-symmetric random Latin Hypercube [1]_.
    
    Data is randomly sampled using a uniform distribution inside each
    bin.
    
    Parameters
    ----------
    m : int
        Number of sampling points.
    d : int
        Number of dimensions.
    
    Returns
    -------
    s : ndarray, shape (m,d,)
        Sampling data. :math:`s \in \mathbb{R}^d : 0 \leq s \leq 1`.
    
    References
    ----------
    .. [1] Helton, J C and F J Davis. 2003. Latin Hypercube Sampling
        and the Propagation of Uncertainty in Analyses of Complex
        Systems. Reliability Engineering & System Safety 81 (1): 23-69.
    """

    bounds = np.zeros((m+1,d,))
    for i in range(d):
        bounds[:,i] = np.linspace(0,1,m+1)

    X = np.random.rand(m,d)
    X = bounds[:-1,:] + X*(bounds[1:,:]-bounds[:-1,:])

    P = np.zeros((m,d),dtype=int)
    P[:,0] = np.arange(m)

    for j in range(1,d):
        P[:,j] = np.random.permutation(np.arange(m))
    
    s = np.zeros((m,d))
    for j in range(d):
        for i in range(m):
            s[i,j] = X[P[i,j],j]
    return s

if __name__ == "__main__":
    print('This is test for LHDstandard')
    dim = 2
    m = 5
    print('dim is ',dim)
    print('m is ',m)
    print('set seed to 5')
    np.random.seed(5)
    for i in range(3):
        print(RLatinHyperCube(m, dim))
