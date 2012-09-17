

.. _example_ensemble_plot_forest_iris.py:


====================================================================
Plot the decision surfaces of ensembles of trees on the iris dataset
====================================================================

Plot the decision surfaces of forests of randomized trees trained on pairs of
features of the iris dataset.

This plot compares the decision surfaces learned by a decision tree classifier
(first column), by a random forest classifier (second column) and by an extra-
trees classifier (third column).

In the first row, the classifiers are built using the sepal width and the sepal
length features only, on the second row using the petal length and sepal length
only, and on the third row using the petal width and the petal length only.



.. image:: images/plot_forest_iris_1.png
    :align: center




**Python source code:** :download:`plot_forest_iris.py <plot_forest_iris.py>`

.. literalinclude:: plot_forest_iris.py
    :lines: 17-
    