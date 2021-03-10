import numpy as np

def RLatinHyperCube(m,d):
    # Latin hypercube sampling
    #   m: number of sample points
    #   d: dimension

    bounds = np.zeros((m+1,d,))
    for i in range(d):
        bounds[:,i] = np.linspace(0,1,m+1)

    X = np.random.rand(m,d)
    X = bounds[:-1,:] + X*(bounds[1:,:]-bounds[:-1,:])

    P = np.zeros((m,d),dtype=int)
    P[:,0] = np.arange(m)
    if m%2 == 0:
        k      = m//2
    else:
        k      = (m-1)//2
        P[k,:] = (k)*np.ones((1,d))

    for j in range(1,d):
        P[0:k,j] = np.random.permutation(np.arange(k))
        for i in range(k):
            if np.random.random() < 0.5:
                P[m-1-i,j] = m-1-P[i,j]
            else:
                P[m-1-i,j] = P[i,j]
                P[i,j]     = m-1-P[i,j]
    IPts = np.zeros((m,d))
    for j in range(d):
        for i in range(m):
            IPts[i,j] = X[P[i,j],j]
    return IPts

if __name__ == "__main__":
    print('This is test for SLHDstandard')
    dim = 2
    m = 5
    print('dim is ',dim)
    print('m is ',m)
    print('set seed to 5')
    np.random.seed(5)
    for i in range(3):
        print(LatinHyperCube(m, dim))
