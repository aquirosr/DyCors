Kernel functions
----------------

Exponential Kernel
~~~~~~~~~~~~~~~~~~
The Exponential Kernel is defined as follows:

.. math::
    \Phi(r) = \exp \left( -\dfrac{r^2}{2 l^2} \right), l>0

.. autofunction:: DyCors.kernels.surrogateRBF_Expo
.. autofunction:: DyCors.kernels.evalRBF_Expo
.. autofunction:: DyCors.kernels.surrogateGRBF_Expo
.. autofunction:: DyCors.kernels.evalGRBF_Expo

Matérn Kernel
~~~~~~~~~~~~~
The half integer simplification of the Matérn Kernel
is defined as follows:

.. math::
    \Phi(r) = \exp \left( -\sqrt{2p+1} \dfrac{r}{l} \right)
    \dfrac{p!}{(2p)!} \sum_{i=0}^{p} \dfrac{(p+i)!}{i!(p-i)!}
    \left( 2 \sqrt{2p+1} \dfrac{r}{l} \right)^{p-i}, p=\nu - 1/2,
    l>0, nu>=1/2

.. autofunction:: DyCors.kernels.surrogateRBF_Matern
.. autofunction:: DyCors.kernels.evalRBF_Matern
.. autofunction:: DyCors.kernels.surrogateGRBF_Matern
.. autofunction:: DyCors.kernels.evalGRBF_Matern

Cubic Kernel
~~~~~~~~~~~~~
The Cubic Kernel is defined as follows:

.. math:: 
    \Phi(r) = r^3

.. autofunction:: DyCors.kernels.surrogateRBF_Cubic
.. autofunction:: DyCors.kernels.evalRBF_Cubic
.. autofunction:: DyCors.kernels.surrogateGRBF_Cubic
.. autofunction:: DyCors.kernels.evalGRBF_Cubic