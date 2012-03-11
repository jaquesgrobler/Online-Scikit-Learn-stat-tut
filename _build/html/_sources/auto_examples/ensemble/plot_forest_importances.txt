

.. _example_ensemble_plot_forest_importances.py:


=========================================
Feature importances with forests of trees
=========================================

This examples shows the use of forests of trees to evaluate the importance of
features on an artifical classification task. The red plots are the feature
importances of each individual tree, and the blue plot is the feature
importance of the whole forest.

As expected, the knee in the blue plot suggests that 3 features are
informative, while the remaining are not.



.. image:: images/plot_forest_importances_1.png
    :align: center


**Script output**::

  Feature ranking:
  1. feature 1 (0.245865)
  2. feature 0 (0.194416)
  3. feature 2 (0.174455)
  4. feature 7 (0.057138)
  5. feature 8 (0.055967)
  6. feature 4 (0.055516)
  7. feature 5 (0.055179)
  8. feature 9 (0.054639)
  9. feature 3 (0.053921)
  10. feature 6 (0.052904)



**Python source code:** :download:`plot_forest_importances.py <plot_forest_importances.py>`

.. literalinclude:: plot_forest_importances.py
    :lines: 14-
    