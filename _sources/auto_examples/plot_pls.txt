

.. _example_plot_pls.py:


=========================
PLS Partial Least Squares
=========================

Simple usage of various PLS flavor:
- PLSCanonical
- PLSRegression, with multivariate response, a.k.a. PLS2
- PLSRegression, with univariate response, a.k.a. PLS1
- CCA

Given 2 multivariate covarying two-dimensional datasets, X, and Y,
PLS extracts the 'directions of covariance', i.e. the components of each
datasets that explain the most shared variance between both datasets.
This is apparent on the **scatterplot matrix** display: components 1 in
dataset X and dataset Y are maximaly correlated (points lie around the
first diagonal). This is also true for components 2 in both dataset,
however, the correlation across datasets for different components is
weak: the point cloud is very spherical.



.. image:: images/plot_pls_1.png
    :align: center


**Script output**::

  Corr(X)
  [[ 1.    0.5  -0.07  0.04]
   [ 0.5   1.    0.07  0.06]
   [-0.07  0.07  1.    0.5 ]
   [ 0.04  0.06  0.5   1.  ]]
  Corr(Y)
  [[ 1.    0.46 -0.04  0.01]
   [ 0.46  1.   -0.04 -0.02]
   [-0.04 -0.04  1.    0.54]
   [ 0.01 -0.02  0.54  1.  ]]
  True B (such that: Y = XB + Err)
  [[1 1 1]
   [2 2 2]
   [0 0 0]
   [0 0 0]
   [0 0 0]
   [0 0 0]
   [0 0 0]
   [0 0 0]
   [0 0 0]
   [0 0 0]]
  Estimated B
  [[ 1.   1.   1. ]
   [ 2.   2.1  2. ]
   [ 0.  -0.   0. ]
   [-0.   0.  -0. ]
   [-0.1  0.  -0. ]
   [ 0.  -0.   0. ]
   [ 0.   0.  -0. ]
   [ 0.   0.   0. ]
   [-0.   0.   0. ]
   [-0.  -0.  -0. ]]
  Estimated betas
  [[ 1.]
   [ 2.]
   [ 0.]
   [-0.]
   [-0.]
   [-0.]
   [-0.]
   [ 0.]
   [ 0.]
   [ 0.]]



**Python source code:** :download:`plot_pls.py <plot_pls.py>`

.. literalinclude:: plot_pls.py
    :lines: 21-
    