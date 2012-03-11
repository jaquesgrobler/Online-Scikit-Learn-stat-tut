

.. _example_linear_model_plot_ridge_path.py:


===========================================================
Plot Ridge coefficients as a function of the regularization
===========================================================

.. currentmodule:: sklearn.linear_model

Shows the effect of collinearity in the coefficients or the
:class:`Ridge`. At the end of the path, as alpha tends toward zero
and the solution tends towards the ordinary least squares, coefficients
exhibit big oscillations.



.. image:: images/plot_ridge_path_1.png
    :align: center




**Python source code:** :download:`plot_ridge_path.py <plot_ridge_path.py>`

.. literalinclude:: plot_ridge_path.py
    :lines: 13-
    