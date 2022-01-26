Kernel functions
----------------
Here we provide a description of the different classes used to build
and evaluate the RBF and the GRBF surrogate models. Three different
kernel functions :math:`\phi(r)` are defined in the code, the Exponential
kernel (:class:`RBF_Exponential <.kernels.exponential.RBF_Exponential>`, 
:class:`GRBF_Exponential <.kernels.exponential.GRBF_Exponential>`),
the Matérn kernel (:class:`RBF_Matern <.kernels.matern.RBF_Matern>`, 
:class:`GRBF_Matern <.kernels.matern.GRBF_Matern>`) and the Cubic kernel
(:class:`RBF_Cubic <.kernels.cubic.RBF_Cubic>`, 
:class:`GRBF_Cubic <.kernels.cubic.GRBF_Cubic>`).

The ``fit`` method of these classes solves the system :math:`A s = F`.
In the case of RBF interpolants, the vector :math:`F` contains the values
of the function where the points have been evaluated and the matrix :math:`A`
is defined as:

.. math::
    A = \begin{bmatrix}
            \Phi & P \\
            P^T & 0.
        \end{bmatrix}

In the case of GRBF interpolants, the vector :math:`F` contains both the values
of the function and its gradient and the matrix :math:`A` is defined as:

.. math::
    A = \begin{bmatrix}
            \Phi & \Phi_\mathrm{d} \\
            -\Phi_\mathrm{d}^T & \Phi_\mathrm{dd},
        \end{bmatrix}

where :math:`\Phi` is the RBF kernel matrix, :math:`P` is a matrix with linear
polynomial terms, :math:`\Phi_\mathrm{d}` is the first derivative of the RBF
kernel matrix :math:`\Phi` and :math:`\Phi_\mathrm{dd}` is the second derivative
[1]_, [2]_.

The matrix :math:`\Phi` is defined as:

.. math::
    \Phi_{i,j} = \phi(r_{i,j}),

where :math:`r_{i,j} = \left \| x^i-x^j \right\|` is the Euclidean distance
between the points :math:`x^i` and :math:`x^j`. Tha matrices :math:`\Phi_\mathrm{d}`
and :math:`\Phi_\mathrm{dd}` are defined as:

.. math::
    \Phi_{\mathrm{d}_{i,j,k}} = \dfrac{\partial \Phi_{i,j}}{\partial x_k^i} 
    = \dfrac{\partial \Phi_{i,j}}{\partial r_{i,j}}
    \dfrac{\partial r_{i,j}}{\partial x_k^i},

.. math::
    \Phi_{\mathrm{dd}_{i,j,k,l}} = \dfrac{\partial \Phi_{\mathrm{d}_{i,j,k}}}{\partial x_l^i}
    = \dfrac{\partial^2 \Phi_{i,j}}{\partial r_{i,j}^2}
    \dfrac{\partial r_{i,j}}{\partial x_l^i}
    \dfrac{\partial r_{i,j}}{\partial x_k^i}
    
    + \dfrac{\partial \Phi_{i,j}}{\partial r_{i,j}}
    \dfrac{\partial^2 r_{i,j}}{\partial (x_k^i) \partial (x_l^i)}.

where :math:`x_k^i` is the `kth` component of the `ith` point in the vector
of evaluated points `x`.

Once the ``fit`` method has been used to build the surrogate model, it is possible
to evaluate points using the method ``evaluate``. If needed, the internal parameters
can be updated using the ``update`` method.

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

where :math:`p=\left \lfloor{\nu - 1/2}\right \rfloor , l>0, \nu>=1/2`
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
    \Phi(r) = r^3/l^3

where :math:`l>0`, :math:`r=\|y-x\|`, where :math:`\|\cdot\|` is the
Euclidean norm.

The first derivative of the Cubic Kernel with respect to the
distance is defined as:

.. math::
    \Phi_{,r}(r) = 3 r^2/l^3

The second derivative is defined as:

.. math::
    \Phi_{,rr}(r) = 6 r/l^3

.. automodule:: DyCors.kernels.cubic
    :members: