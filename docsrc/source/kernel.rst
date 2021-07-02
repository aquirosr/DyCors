Kernel functions
----------------
Here we provide a description of the different functions used to build
and evaluate the RBF and the GRBF.

The full RBF matrix A is defined as:

.. math::
    A = \begin{bmatrix}
            \Phi & P \\
            P^T & 0
        \end{bmatrix}

where :math:`\Phi` is the RBF kernel matrix and :math:`P` is the matrix
with the linear polynomial terms [1]_.

The full GRBF matrix A is defined as:

.. math::
    A = \begin{bmatrix}
            \Phi & \Phi_d \\
            -\Phi_d^T & \Phi_{dd}
        \end{bmatrix}

where :math:`\Phi_d` is the first derivative of the RBF kernel matrix
:math:`\Phi` and :math:`\Phi_{dd}` is the second derivative [2]_. The
matrices :math:`\Phi_d` and :math:`\Phi_{dd}` are defined as:

.. math::
    \Phi_d = \dfrac{\partial \Phi}{\partial r}
    \dfrac{\mathrm{d} r}{\mathrm{d} x_k^i},

.. math::
    \Phi_{dd} = \dfrac{\partial \Phi_d}{\partial x_k^i} =
    \dfrac{\partial^2 \Phi}{\partial r^2}
    \left( \dfrac{\mathrm{d} r}{\mathrm{d} x_k^i} \right)^2
    + \dfrac{\partial \Phi}{\partial r}
    \dfrac{\mathrm{d}^2 r}{\mathrm{d} (x_k^i)^2},

where :math:`r` is the Euclidean distance and :math:`x_k^i` is the
`kth` component of the `ith` point in the vector of points `x`.

.. [1] Regis, R G and C A Shoemaker. 2013. Combining radial basis
        function surrogates and dynamic coordinate search in
        high-dimensional expensive black-box optimization. Engineering
        Optimization 45 (5): 529-555.
.. [2] Laurent, L, R Le Riche, B Soulier and P A Boucard. 2019. An
    Overview of Gradient-Enhanced Metamodels with Applications.
    Archives of Computational Methods in Engineering 26 (1). 61-106. 

Exponential Kernel
~~~~~~~~~~~~~~~~~~
The Exponential Kernel is defined as follows:

.. math::
    \Phi(r) = \exp \left( -\dfrac{r^2}{2 l^2} \right),
    
where :math:`l>0` and :math:`r=\|y-x\|`, where :math:`\|\cdot\|` is the
Euclidean norm.

The first derivative of the Exponential Kernel with respect to the
distance is defined as:

.. math::
    \Phi_{,r}(r) = - \dfrac{r}{l^2} \exp \left( -\dfrac{r^2}{2 l^2} \right)

The second derivative is defined as:

.. math::
    \Phi_{,rr}(r) = - \dfrac{1}{l^2} \exp \left( -\dfrac{r^2}{2 l^2} \right)
    + \dfrac{r^2}{l^4} \exp \left( -\dfrac{r^2}{2 l^2} \right)

.. automodule:: DyCors.kernels.exponential
    :members:

Matérn Kernel
~~~~~~~~~~~~~
The half integer simplification of the Matérn Kernel
is defined as follows:

.. math::
    \Phi(r) = \exp \left( -\sqrt{2p+1} \dfrac{r}{l} \right)
    \dfrac{p!}{(2p)!} \sum_{i=0}^{p} \dfrac{(p+i)!}{i!(p-i)!}
    \left( 2 \sqrt{2p+1} \dfrac{r}{l} \right)^{p-i},

where :math:`p=\left \lfloor{\nu - 1/2}\right \rfloor , l>0, nu>=1/2`
and :math:`r=\|y-x\|`, where :math:`\|\cdot\|` is the Euclidean norm.

The first derivative of the Matérn Kernel with respect to the
distance is defined as:

.. math::
    \Phi_{,r}(r) = - \dfrac{\sqrt{2p+1}}{l}
    \exp \left( -\sqrt{2p+1} \dfrac{r}{l} \right) \dfrac{p!}{(2p)!}
    \sum_{i=0}^{p} \dfrac{(p+i)!}{i!(p-i)!}
    \left( 2 \sqrt{2p+1} \dfrac{r}{l} \right)^{p-i} \\
    + \dfrac{2 \sqrt{2p+1}}{l} 
    \exp \left( -\sqrt{2p+1} \dfrac{r}{l} \right)
     \dfrac{p!}{(2p)!} \sum_{i=0}^{p-1} \dfrac{(p+i)!}{i!(p-i)!}
    \left( 2 \sqrt{2p+1} \dfrac{r}{l} \right)^{p-i-1}

The second derivative is defined as:

.. math::
    \Phi_{,rr}(r) = \dfrac{2p+1}{l^2}
    \exp \left( -\sqrt{2p+1} \dfrac{r}{l} \right) \dfrac{p!}{(2p)!}
    \sum_{i=0}^{p} \dfrac{(p+i)!}{i!(p-i)!}
    \left( 2 \sqrt{2p+1} \dfrac{r}{l} \right)^{p-i} \\
    - \dfrac{4 (2p+1)}{l^2} 
    \exp \left( -\sqrt{2p+1} \dfrac{r}{l} \right)
     \dfrac{p!}{(2p)!} \sum_{i=0}^{p-1} \dfrac{(p+i)!}{i!(p-i)!}
    \left( 2 \sqrt{2p+1} \dfrac{r}{l} \right)^{p-i-1} \\
    + \dfrac{4 (2p+1)}{l^2} 
    \exp \left( -\sqrt{2p+1} \dfrac{r}{l} \right)
     \dfrac{p!}{(2p)!} \sum_{i=0}^{p-2} \dfrac{(p+i)!}{i!(p-i)!}
    \left( 2 \sqrt{2p+1} \dfrac{r}{l} \right)^{p-i-2}

.. automodule:: DyCors.kernels.matern
    :members:

Cubic Kernel
~~~~~~~~~~~~~
The Cubic Kernel is defined as follows:

.. math:: 
    \Phi(r) = r^3

where :math:`r=\|y-x\|`, where :math:`\|\cdot\|` is the
Euclidean norm.

The first derivative of the Cubic Kernel with respect to the
distance is defined as:

.. math::
    \Phi_{,r}(r) = 3 r^2

The second derivative is defined as:

.. math::
    \Phi_{,rr}(r) = 6 r

.. automodule:: DyCors.kernels.cubic
    :members: