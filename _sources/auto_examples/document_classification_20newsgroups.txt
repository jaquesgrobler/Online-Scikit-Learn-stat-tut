

.. _example_document_classification_20newsgroups.py:


======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how the scikit-learn can be used to classify
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays
and demos various classifiers that can efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset which will be
automatically downloaded and then cached.

You can adjust the number of categories by giving their names to the dataset
loader or setting them to None to get the 20 of them.



**Python source code:** :download:`document_classification_20newsgroups.py <document_classification_20newsgroups.py>`

.. literalinclude:: document_classification_20newsgroups.py
    :lines: 18-
    