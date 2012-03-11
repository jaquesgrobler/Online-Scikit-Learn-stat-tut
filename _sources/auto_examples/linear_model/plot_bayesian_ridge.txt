

.. _example_linear_model_plot_bayesian_ridge.py:


=========================
Bayesian Ridge Regression
=========================

Computes a :ref:`bayesian_ridge_regression` on a synthetic dataset.

Compared to the OLS (ordinary least squares) estimator, the coefficient
weights are slightly shifted toward zeros, wich stabilises them.

As the prior on the weights is a Gaussian prior, the histogram of the
estimated weights is Gaussian.

The estimation of the model is done by iteratively maximizing the
marginal log-likelihood of the observations.



.. rst-class:: horizontal


    *

      .. image:: images/plot_bayesian_ridge_1.png
            :scale: 47

    *

      .. image:: images/plot_bayesian_ridge_3.png
            :scale: 47

    *

      .. image:: images/plot_bayesian_ridge_2.png
            :scale: 47




**Python source code:** :download:`plot_bayesian_ridge.py <plot_bayesian_ridge.py>`

.. literalinclude:: plot_bayesian_ridge.py
    :lines: 17-
    