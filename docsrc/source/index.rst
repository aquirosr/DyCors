.. DyCors documentation master file, created by
   sphinx-quickstart on Fri Apr  2 14:04:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Reference documentation
=======================

DyCors/G-DyCors
---------------

DYCORS (DYnamic COordinate search using Response Surfaces),
G-DYCORS (Gradient-enhanced DYnamic COordinate search using Response Surfaces).

Version implemented here is DYCORS-LMSRBF (Local Metric Stochastic
Radial Basis Functions) [1]_. 

Both RBF (Radial Basis Functions) and GRBF (Gradient-enhanced Radial
Basis Functions) can be used. Three different kernels have been
implemented: exponential kernel, Matérn kernel and cubic kernel. The internal
parameters of the kernel can be optimized.

Source code available in `DyCors code
<https://github.com/aquirosr/DyCors>`_.

Installation
------------
Follow this commands to install DyCors::

   git clone https://github.com/aquirosr/DyCors.git
   cd DyCors
   pip install -e .

API Reference
-------------

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   minimize
   tools

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`search`

References
----------

.. [1] Regis, R G and C A Shoemaker. 2013. Combining radial basis
   function surrogates and dynamic coordinate search in
   high-dimensional expensive black-box optimization. Engineering
   Optimization 45 (5): 529-555.