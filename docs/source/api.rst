Reference Documentation
=======================

.. note::

    Currently only few links to classes do work (likely my failure to master the 
    excellent `sphinx <https://www.sphinx-doc.org/en/master/>`_ tool). So please use the navigation bar to the left.

.. toctree::
   :hidden:
   :maxdepth: 1
   :glob:

   stubs/mymod.kmeanstf.KMeansTF
   stubs/mymod.kmeanstf.TunnelKMeansTF
   stubs/mymod.kmeanstf.BaseKMeansTF

kmeanstf consists of two classes

* :class:`.KMeansTF` (close equivalent of scikit-learn KMeans)
* :class:`.TunnelKMeansTF` (implements tunnel k-means)

which share a common base class

* :class:`BaseKMeansTF`

This design was chosen to 

* enable :class:`.KMeansTF` to have an interface very similar to `scikit-learns KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_
* avoid code duplication





