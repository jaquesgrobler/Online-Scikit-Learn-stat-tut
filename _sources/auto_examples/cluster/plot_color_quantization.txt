

.. _example_cluster_plot_color_quantization.py:


==================================
Color Quantization using K-Means
==================================

Performs a pixel-wise Vector Quantization (VQ) of an image of the summer palace
(China), reducing the number of colors required to show the image from 96,615
unique colors to 64, while preserving the overall appearance quality.

In this example, pixels are represented in a 3D-space and K-means is used to
find 64 color clusters. In the image processing literature, the codebook
obtained from K-means (the cluster centers) is called the color palette. Using
a single byte, up to 256 colors can be addressed, whereas an RGB encoding
requires 3 bytes per pixel. The GIF file format, for example, uses such a
palette.

For comparison, a quantized image using a random codebook (colors picked up
randomly) is also shown.



.. rst-class:: horizontal


    *

      .. image:: images/plot_color_quantization_1.png
            :scale: 47

    *

      .. image:: images/plot_color_quantization_3.png
            :scale: 47

    *

      .. image:: images/plot_color_quantization_2.png
            :scale: 47


**Script output**::

  Fitting estimator on a small sub-sample of the data
  done in 0.753s.
  Predicting color indices on the full image (k-means)
  done in 0.470s.
  Predicting color indices on the full image (random)
  done in 0.471s.



**Python source code:** :download:`plot_color_quantization.py <plot_color_quantization.py>`

.. literalinclude:: plot_color_quantization.py
    :lines: 21-
    