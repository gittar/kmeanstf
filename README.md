# kmeanstf
## k-means++ implementation based on tensorflow-gpu
![alt text][logo]

[logo]: https://raw.githubusercontent.com/gittar/kmeanstf/master/img/million_100.png "k-means++ example"
# Quick Start

To use k-means++ with GPU-support do the following:

```pip install kmeanstf```

(requires tensorflow-gpu, at least version 1.14 or 2.0b)

Execute the following [test program](https://github.com/gittar/kmeanstf/blob/master/demo/test.py) to produce the above graphic.

```python
import tensorflow as tf
import matplotlib.pyplot as plt
from kmeanstf import KMeansTF

# create data set
X = tf.random.normal([50000,2])
# create kmeanstf object
km = KMeansTF(n_clusters=100, n_init=1)
# adapt
km.fit(X)

# plot result (optional)
m=10000 # max number of data points to display
fig,ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:m,0],X[:m,1],s=1)
ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=8,c="r")
plt.show()
```




# What is k-means?
[k-means](https://en.wikipedia.org/wiki/K-means_clustering) - a.k.a. Lloyds Algorithm - is perhaps the most well-known clustering method. It positions k centroids (means) over a given data set in order to represent/cluster the data set.

Starting from some initial positions of the centroids a number of so-called "Lloyd iterations" is performed until a stopping criterion is met. One Lloyd iteration consists of the following two steps:
* determine for each centroid the subset of data points for which this centroid is closest
* move each centroid to the mean of its associated data sub-set (hence the name "k-means")

It can be shown that each Lloyd iteration decreases the sum of squared Euclidean distances of all data points to their respective nearest centroid or leaves it unchanged in which case the algorithm has converged.

The final sum of distances depends strongly on the initial centroid positions. A common goal is to find a set of centroids with a small sum of distances (see example below). 


# k-means++
In 2006 a particular initialization method was proposed which produces provably good (but generally still sub-optimal) results: [k-means++](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf). One can describe the core of k-means++ as the sequential positioning of centroids in areas where the given data points are far from the already positioned centroids. k-means++ is said to both generate better solutions than k-means with random initialization as well as needing fewer Lloyd iterations.

Probably due to its good results k-means++ has been chosen as the default initialization method for the implementation of k-means in the popular [scikit-learn  python library](https://scikit-learn.org)  for machine learning

# why kmeanstf?

k-means for large data sets and large number of centroids is computationally expensive (slow!). For portability reasons scikit-learn does not support the use of GPUs (https://scikit-learn.org/stable/faq.html#will-you-add-gpu-support). Therefore we provide here an implementation of k-means (and the k-means++ initialization) based on tensorflow and making use of a GPU if available. The k-means++ initialization is a port from the [scikit-learn implementation of k-means++](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/k_means_.py) to tensorflow (using tensorflow matrix operations instead of the numpy ones used by scikit-learn). The tensorflow-based implementation of Lloyd iterations borrows some aspects from 
Shawn Simister's gist https://gist.github.com/narphorium/d06b7ed234287e319f18 and added the splitting of large data sets into subsets of a given size as well as finding a suitable size automatically for a given GPU if memory errors occur with the default size.

**Please note:** Only a subset of the scikit-learn KMeans class is implemented in KMeansTF, in particular the fit(), predict() and fit_predict() methods. There is e.g. no handling of sparse matrices, no mini-batch version, no sample weights. For these features please use scikit-learn. Using kmeanstf makes most sense if you like to perform standard k-means (k-means++) on a GPU for large data sets.
# Speed-up

On a linux machine (Ubuntu 18.04 LTS) equipped with a NVidia GTX-1060 6MB graphics card we observed a speed-up of 7-10 for larger data sets (n > 200k points). Example: for a data set of n = 10⁶ (1 million) 2-dimensional vectors and k=100 centroids the execution time for 30 Lloyd iterations was 6.34 seconds a.o.t. 62.17 seconds for scikit-learn (this is a speed-up of 9.81)

![alt text][barspeed]

[barspeed]: https://raw.githubusercontent.com/gittar/kmeanstf/master/img/barspeed3.png "execution times"

Below you see speed-up values measured for different values of data set size *n* and number of centroids *k* for 2D-data. For larger data sets also the speed-up tends to be higher. For small data sets, however, KMeansTF is actually often slower than scikit-learn KMeans. Perhaps this is caused caused by a start-up time for tensorflow. One could argue that this is not so relevant but it may also indicate potential for improvement for the kmeanstf implementation.
![alt text][speed-up]


[speed-up]: https://raw.githubusercontent.com/gittar/kmeanstf/master/img/speedupa2.png "speed-up factors"

# Why is it so fast?

Three lines of code which seem to originate from Shawn Simister (https://twitter.com/narphorium/status/668234313662525440?lang=en) are at the core of the tensorflow k-means implementation:

```python
expanded_vectors = tf.expand_dims(samples, 0) #[1,n,d]
expanded_centroids = tf.expand_dims(centroids, 1) # [k,1,d]
distances = tf.reduce_sum( tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2) #[k,n]
```
Thereby **samples** is the [*n,d*] tensor containing *n* vectors of dimension *d*.

**centroids** is the [*k,d*] tensor containing *k* vectors (centroids) of dimension *d*.

The first two lines add a dimension of size 1 to samples and centroids in position 0 and 1 respectively

The third line contains the inner statement
```python
tf.subtract(expanded_vectors, expanded_centroids) #[k,n,d]
```
This makes use of "broadcasting", an operation in tensorflow and numpy which aligns the shape of tensors before an arithmetic operation is applied which requires them to have the same shape. In this case the [1,n,d]-tensor **expanded_vectors** is converted to shape [*k,n,d*] by stacking *k* copies of **samples** along dimension 0. The [*k,1,d*] tensor **expanded_centroids** is converted to shape [*k,n,d*] as well by stacking *n* copies of **centroids** along dimension 1. 

This makes it possible to compute the elementwise mutual distance of all *n* data vectors and all *k* centroids. Adding the outer tf.square and tf.reduce_sum operations we have one single tensor statement which does the work of 4 nested conventional loops. 

Pure elegance! 

And ideally suited to run on a GPU (if it has enough memory, because *k\*n\*d* can be quite large). To not make the GPU memory a limiting factor, the implementation in kmeanstf is able to split up this tensor along the *n*-dimension into several pieces each fitting on the GPU. See parameter **max_mem** below.

# class KMeansTF

**Please note:**  the interface is designed to be a subset of [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans).  
Current exception: parameter **max_mem** (see below)

## class kmeanstf.KMeansTF(n_clusters=8, init='k-means++', n_init=10,                 max_iter=300, tol=1e-4, verbose=0, max_mem=1300000000)
## Parameters
* **n_clusters : int, optional, default: 8**

    The number of clusters to form as well as the number of centroids to generate.
* **init : {‘k-means++’, ‘random’ or an ndarray}**

    Method for initialization, defaults to ‘k-means++’:

    ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.

    ‘random’: choose k observations (rows) at random from data for the initial centroids.

    If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

* **n_init : int, default: 10**

    Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
* **max_iter : int, default: 300**

    Maximum number of iterations of the k-means algorithm for a single run.
* **tol : float, default: 1e-4**

    Relative tolerance with regards to inertia to declare convergence
* **verbose : int, default: 0**
    if larger 0 outputs messages to analyze the working of the algorithm

* **max_mem : int, default: 1300000000**
    (no equivalent in sklearn) size in bytes of largest tensor to be used during k-means computation. Default value is suitable for NVidia GTX-1060. If a too large value is specified and a memory error occurs, a sequence of decreasing values is automatically tried out (and printed) until a working value for the given GPU is found. This value should subsequently be used as the **ḿax_mem** parameter. Large data sets ar split into fractions in order to not pass this threshold.

## Attributes

* **cluster_centers_ : array, [n_clusters, n_features]**

    Coordinates of cluster centers. 

* **inertia_ : float**

    Sum of squared distances of samples to their closest cluster center.
* **n_iter_ : int**

    Number of iterations run.

## Methods

* **fit (self, X)**

    Compute k-means clustering.

* **fit_predict (self, X)**

    Compute cluster centers and predict cluster index for each sample.

    **Parameters**
    X: data to be clustered and predicted, shape = [n_samples, n_features]

    **Returns:**
    labels : array, shape [n_samples,]

    Index of the cluster each sample belongs to

* **predict (self, X)**    

    Predict cluster index for each sample.

    **Parameters**
    X: data to be predicted, shape = [n_samples, n_features]

    **Returns:**
    labels : array, shape [n_samples,]

    Index of the cluster each sample belongs to
# Areas for further improvements

* improve the time behavior for smaller data sets
* directly determine suitable max_mem value from physical features of present GPU
* enable the use of several GPUs
* enable the use of TPUs (kmeanstf does work on colab with GPU, but not yet with TPU)

Both feedback and pull requests are welcome.

