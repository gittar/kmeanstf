Getting Started
===============

Installation
------------

kmeanstf is available from pypi and can thus be installed using `pip <https://pypi.org/project/pip/>`_:

.. code:: shell

    $ pip install kmeanstf

Self Test
---------

kmeanstf has a simple self-testing/demo routine which is realized as static member :meth:`.self_test`. 
Per default it generates a random data set from a mixture of Gaussians and runs both *k*-means++ (from scikit-learn)
and tunnel *k*-means. The summed squared error (SSE) and the runtime is printed out and the results are
plotted via matplotlib.

If you execute the following code (with python 3)

.. literalinclude:: examples/selftest.py

a table and a plot are generated:

.. code-block:: shell

    self test ...
    Data is mixture of 50 Gaussians in unit square with sigma=0.00711
    algorithm      | data.shape  |   k  | init      | n_init  |     SSE   | Runtime  | Improvement
    ---------------|-------------|------|-----------|---------|-----------|----------|------------
    k-means++      | (10000, 2)  |  100 | k-means++ |      10 |   0.66363 |    2.23s | 0.00%
    tunnel k-means | (10000, 2)  |  100 | random    |       1 |   0.64342 |    2.81s | 3.05%

.. image:: ./img/selftest.png
    :width: 700px
    :align: center
    :alt: alternate text

Since in this problem the number of centroids (100) is twice as high as the number of Gaussian clusters (50), 
it seems reasonable to position two centroids in each Gaussian cluster. One can note that *k*-means++ leaves 
a number of clusters with only one centroid which leads to a higher SSE than tunnel k-means.

The :meth:`.self_test` method can also be parametrized to have a different mixture of Gaussians (e.g. only 20 Gaussians) 
or a different value of *n_clusters* (a.k.a. *k*). Also an own data set can be provided in parameter *X* (see documentation).

.. _kmp-label:

Using KMeansTF
--------------

The class :class:`.KMeansTF` is a very close equivalent of  scikit-learn's `KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ class.

Executing the following code

.. literalinclude:: examples/exrand.py

.. note::

    The parameter ``random_state`` was set to 1 here to seed the random number generators of python, numpy and tensorflow. 
    Thus the generated data set is identical to the following example and the SSE values can be compared.

leads to:

.. image:: ./img/kmpp1.png
    :width: 500px
    :align: center
    :alt: alternate text


Using TunnelKMeansTF
--------------------

The class :class:`.TunnelKMeansTF` implements the tunnel *k*-means algorithm as described 
in `arXiv:1706.09059 <https://arxiv.org/abs/1706.09059>`_. Its API is very close to scikit-learn's `KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ class 
but has some additional optional parameters related to tunnel *k*-means.
 
Executing the  code

.. literalinclude:: examples/exrand2.py

leads to:

.. image:: ./img/tkm1.png
    :width: 500px
    :align: center
    :alt: alternate text

The data set used is the same as in :ref:`kmp-label` for *k*-means++. One can note that the SSE of the tunnel *k*-means solution is lower.

Optimizing GPU-Usage
====================

If a GPU is used (kmeanstf does work also without one, just slower) certain 
optimizations are possible depending on the size of the GPU memory. These are described below.


Both :class:`.KMeansTF` and :class:`.TunnelKMeansTF` have a parameter ``max_mem``. 
This parameter is used to decide if for a given *k*-means problem characterized by data set *X[n,d]* and number of 
centroids (parameter *cluster_centers*, a.k.a. *k*) the central distance computation (distance of all centroids to all data points) can be 
performed in one step or has to be done with subsets of the data. For the default data type ``tf.float32`` (4 bytes) this distance computation requires a matrix of size 

.. math::

    S = n \times d \times k \times 4

Therefore, if 

.. math::

    S > \mbox{max_mem}
    
the data set is split into sufficiently many parts that the above matrix for each part has at most size ``max_mem`` and 
the distances for all parts are computed separately. 

In case of a 
GPU memory error (which may occur depending on the combination of ``max_mem`` value and actual size of the GPU memory) the value of ``max_mem`` is 
reduced by a specific fraction (currently 10%) until a suitable value is found.

This default value  (1_800_000_000) for ``max_mem`` has been empirically selected for a particular graphics card (NVidia GTX-1060 6GB). 
It is considerably lower than the available GPU memory (6GB in this case) since also for other computations 
some GPU-memory is required. 

For a much larger or smaller GPU-memory it might make sense to provide a non-default value for the ``max_mem`` parameter. By setting the parameter ``verbose`` to 1 for 
:class:`.KMeansTF` or :class:`.TunnelKMeansTF` one can observe messages about splitting the data set or about GPU-errors and can find a suitable value for the graphics card at hand.



Differences between kmeanstf and scikit-learns's KMeans class
=============================================================

The following performance-related functionalities of scikit-learn's 
`KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ class 
were not implemented in :class:`.KMeansTF` and :class:`.TunnelKMeansTF` since kmeanstf provides data parallelism via GPU usage:

* elkan algorithm (parameter *algorithm*, value 'elkan')
* support for job-level parallelism (parameter *n_jobs*)
* precomputing distances (parameters *precompute_distances*, *copy_x*)



