

.. _example_plot_classification_probability.py:


===============================
Plot classification probability
===============================

Plot the classification probability for different classifiers. We use a 3
class dataset, and we classify it with a Support Vector classifier, as
well as L1 and L2 penalized logistic regression.

The logistic regression is not a multiclass classifier out of the box. As
a result it can identify only the first class.



.. image:: images/plot_classification_probability_1.png
    :align: center


**Script output**::

  classif_rate for Linear SVC : 76.000000 
  classif_rate for L1 logistic : 33.333333 
  classif_rate for L2 logistic : 56.666667



**Python source code:** :download:`plot_classification_probability.py <plot_classification_probability.py>`

.. literalinclude:: plot_classification_probability.py
    :lines: 13-
    