Further Topics
==============


Differences between kmeanstf and scikit-learns's KMeans class
-------------------------------------------------------------

The following parameters and the related functionality of scikit-learn's 
`KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ class 
are curently not implemented in kmeanstf:

* ``algorithm`` (makes it possible to choose 'full' or 'elkan')in elkan algorithm (parameter *algorithm*, value 'elkan')
* ``n_jobs`` (support for job-level parallelism)
* ``precompute_distances`` (support for precomputing distances)
* ``copy_x`` (support for precomputing distances)
