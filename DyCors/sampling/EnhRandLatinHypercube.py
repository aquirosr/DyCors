import numpy as np

def ERLatinHyperCube(m, d):
    """Non-symmetric enhanced random Latin Hypercube [1]_.
    
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
    .. [1] Beachkofski, B and R Grandhi. 2002. Improved Distributed
        Hypercube Sampling. 43rd AIAA/ASME/ASCE/AHS/ASC Structures,
        Structural Dynamics, and Materials Conference. Denver,
        Colorado.
    """
    # optimum distance between sample points
    d_opt = m / m**(1/d)
    
    # duplication factor
    dup_factor = 8

    bounds = np.zeros((m+1,d,))
    for i in range(d):
        bounds[:,i] = np.linspace(0,1,m+1)

    X = np.random.rand(m,d)
    X = bounds[:-1,:] + X*(bounds[1:,:]-bounds[:-1,:])

    P = np.zeros((m,d), dtype=int)

    # remainings bins in each dimension
    rem_bins = []
    for i in range(d):
        rem_bins.append(np.arange(m).tolist())
    
    # pick 1st point randomly
    P0 = []
    for i in range(d):
        P0.append(np.random.permutation(np.arange(m))[0])
        
    P[0,:] = np.asarray(P0)
    
    for i in range(m-1):
        nr_bins = m-i-1
        # remove last bin from remaining bins
        for j in range(d):
            rem_bins[j].remove(P[i,j])
        
        # create array with all possible coordinates.
        # If number of possibilities is too large,
        # compute a subspace
        if nr_bins**d<=(dup_factor-1)**(dup_factor-1) and d<=32:
            coords = np.reshape( np.asarray(np.meshgrid(*[np.asarray(dimk)
                                                        for dimk in rem_bins])),
                                (d,nr_bins**d,) )
        else:
            coords = np.zeros((d,nr_bins*dup_factor,))
            for j in range(d):
                coords[j,:] = np.random.choice(rem_bins[j], size=nr_bins*dup_factor)
        
        # compute distances from all possible points to all selected points
        dist = np.linalg.norm(coords[np.newaxis,...] - P[:i+1,:,np.newaxis], 
                              axis=1)
        
        # compute the minimum distance for each possible bin
        id_dist_min = np.argmin(dist, axis=0)
        dist_min = np.zeros(dist.shape[-1])
        for j,idx in enumerate(id_dist_min):
            dist_min[j] = dist[idx,j]
        
        # pick the index of the bin where the distance
        # is closer to the optimum
        dist = np.abs(dist_min - d_opt)
        idx = np.where(dist==dist.min())[-1][0]
        
        P[i+1,:] = coords[:,idx]

    s = np.zeros((m,d))
    for j in range(d):
        for i in range(m):
            s[i,j] = X[P[i,j],j]
    return s

if __name__ == "__main__":
    print('This is test for enhanced LHD')
    dim = 2
    m = 5
    print('dim is ',dim)
    print('m is ',m)
    print('set seed to 5')
    np.random.seed(5)
    for i in range(3):
        print(ERLatinHyperCube(m, dim))
