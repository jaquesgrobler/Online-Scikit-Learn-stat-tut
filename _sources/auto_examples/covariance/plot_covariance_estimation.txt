

.. _example_covariance_plot_covariance_estimation.py:


===========================================
Ledoit-Wolf vs Covariance simple estimation
===========================================

The usual covariance maximum likelihood estimate can be regularized
using shrinkage. Ledoit and Wolf proposed a close formula to compute
the asymptotical optimal shrinkage parameter (minimizing a MSE
criterion), yielding the Ledoit-Wolf covariance estimate.

Chen et al. proposed an improvement of the Ledoit-Wolf shrinkage
parameter, the OAS coefficient, whose convergence is significantly
better under the assumption that the data are gaussian.

In this example, we compute the likelihood of unseen data for
different values of the shrinkage parameter, highlighting the LW and
OAS estimates. The Ledoit-Wolf estimate stays close to the likelihood
criterion optimal value, which is an artifact of the method since it
is asymptotic and we are working with a small number of observations.
The OAS estimate deviates from the likelihood criterion optimal value
but better approximate the MSE optimal value, especially for a small
number a observations.




.. image:: images/plot_covariance_estimation_1.png
    :align: center




**Python source code:** :download:`plot_covariance_estimation.py <plot_covariance_estimation.py>`

.. literalinclude:: plot_covariance_estimation.py
    :lines: 25-
    