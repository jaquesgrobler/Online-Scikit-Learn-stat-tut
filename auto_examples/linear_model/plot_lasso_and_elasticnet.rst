

.. _example_linear_model_plot_lasso_and_elasticnet.py:


========================================
Lasso and Elastic Net for Sparse Signals
========================================




.. image:: images/plot_lasso_and_elasticnet_1.png
    :align: center


**Script output**::

  Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
     normalize=False, precompute=auto, tol=0.0001, warm_start=False)
  r^2 on test data : 0.384710
  ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
        normalize=False, precompute=auto, rho=0.7, tol=0.0001,
        warm_start=False)
  r^2 on test data : 0.240176



**Python source code:** :download:`plot_lasso_and_elasticnet.py <plot_lasso_and_elasticnet.py>`

.. literalinclude:: plot_lasso_and_elasticnet.py
    :lines: 7-
    