

.. _example_svm_plot_svm_parameters_selection.py:


========================================================
Seleting hyper-parameter C and gamma of a RBF-Kernel SVM
========================================================

For SVMs, in particular kernelized SVMs, setting the hyperparameter
is crucial but non-trivial.
In practice, they are usually set using a hold-out validation
set or using cross validation.

This example shows how to use stratified K-fold crossvalidation
to set C and gamma in an RBF-Kernel SVM.

We use a logarithmic grid for both parameters.



.. image:: images/plot_svm_parameters_selection_1.png
    :align: center


**Script output**::

  ('The best classifier is: ', SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
    gamma=0.10000000000000001, kernel='rbf', probability=False, scale_C=True,
    shrinking=True, tol=0.001))



**Python source code:** :download:`plot_svm_parameters_selection.py <plot_svm_parameters_selection.py>`

.. literalinclude:: plot_svm_parameters_selection.py
    :lines: 16-
    