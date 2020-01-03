Further Topics
==============

Optimizing GPU-Usage
--------------------

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
-------------------------------------------------------------

The following performance-related functionalities of scikit-learn's 
`KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ class 
were not implemented in :class:`.KMeansTF` and :class:`.TunnelKMeansTF` since kmeanstf provides data parallelism via GPU usage:

* elkan algorithm (parameter *algorithm*, value 'elkan')
* support for job-level parallelism (parameter *n_jobs*)
* precomputing distances (parameters *precompute_distances*, *copy_x*)
