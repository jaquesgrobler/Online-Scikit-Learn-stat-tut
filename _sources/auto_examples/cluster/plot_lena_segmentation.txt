

.. _example_cluster_plot_lena_segmentation.py:


=========================================
Segmenting the picture of Lena in regions
=========================================

This example uses :ref:`spectral_clustering` on a graph created from
voxel-to-voxel difference on an image to break this image into multiple
partly-homogenous regions.

This procedure (spectral clustering on an image) is an efficient
approximate solution for finding normalized graph cuts.



.. image:: images/plot_lena_segmentation_1.png
    :align: center




**Python source code:** :download:`plot_lena_segmentation.py <plot_lena_segmentation.py>`

.. literalinclude:: plot_lena_segmentation.py
    :lines: 13-
    