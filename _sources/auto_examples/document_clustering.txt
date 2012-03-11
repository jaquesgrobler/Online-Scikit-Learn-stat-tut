

.. _example_document_clustering.py:


=======================================
Clustering text documents using k-means
=======================================

This is an example showing how the scikit-learn can be used to cluster
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

Two algorithms are demoed: ordinary k-means and its faster cousin minibatch
k-means.



**Python source code:** :download:`document_clustering.py <document_clustering.py>`

.. literalinclude:: document_clustering.py
    :lines: 14-
    