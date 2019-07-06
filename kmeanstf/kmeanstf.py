import tensorflow as tf
import numpy as np
import math
import os
import sys
from time import time
from .initializer import Initializer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

class MyProblem(Exception):
    pass

class KMeansTF:
    """ k-means++ using tensorflow
    the interface is taken from sklearn
    evaluated options:
    * n_clusters
    * init
    * n_init
    * max_iter
    * tol (interpreted differently: iteration stops if (sse_new-sse_old)/sse_old < tol
      sk_learn looks at changes of centroids
    """
    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, verbose=0, max_mem=1300000000):
        """from scikit learn"""
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.n_init = n_init
        #
        # self.maxmen is an empirically found number to indicate when the 3D-matrix used for distance computation
        # should be partitioned to avoid memory overflow of the GPU
        # the default value is emprically found for NVidia GTX-1060 6MB  
        # may need to be adapted for other GPUs
        # if memory allocation errors occur, a sequence of smaller and smaller max_mem value is tried 
        # and printed until the computation is successfull. 
        # This makes it possible to find the best value for the GPU in use
        # successfully used value for colab GPU: 4000000000
        self.max_mem =  max_mem
        # must work with tensorflow 1.4 and tensorflow 2.0b
        # in some places different code is needed
        self.TF2 = tf.__version__[0]=="2"

    def _tolerance(self, X, tol):
        """Return a tolerance which is independent of the dataset"""
        _, variances = tf.nn.moments(X,axes=[0])
        return tf.math.reduce_mean(variances) * tol

    def fit(self, X):
        """actually compute the clustering for data set X
        """
        self.cluster_centers_, self.inertia_, self.n_iter_, self.init_duration_ = \
            self._k_means(
                X, n_clusters=self.n_clusters,
                init=self.init, n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol)
        return self
    def predict(self, X):
        """return index of nearest centroid for each point in X
        """
        _, nearest = self._get_sse_and_nearest(X,self.cluster_centers_)
        return nearest
    def fit_predict(self,X):
        """convenience function for calling fit() followed by predict()
        """
        self.fit(X)
        return self.predict(X)   
    def squared_norm(self,X):
        return tf.reduce_sum(tf.square(tf.norm(X)))

    def _k_means(self,X, n_clusters, init='k-means++',
            n_init=10, max_iter=300,
            tol=1e-4):
        """K-means clustering algorithm."""
        # initialize
        sse_min = None
        best_n_iter = 0
        for n_i in range(self.n_init):
            start = time()
            if isinstance(self.init, str):
                if self.init == "random":
                    current_centroids, _ =Initializer.init(X,n_clusters,"random")
                elif self.init == "k-means++":
                    current_centroids, _ =Initializer.init(X,n_clusters,"k-means++")
                else:
                    raise MyProblem("not valid as init value: "+self.init)

            else: 
                # assert array
                # isinstance(self.init,(list, np.ndarray))
                current_centroids = self.init
            end = time()
            init_duration = end-start

            # compute X-dependent tolerance value
            tol = self._tolerance(X, tol)
            #
            # Lloyd Iterations
            #
            sse = -1
            for i in range(max_iter):
                # create copy to measure tol
                centroids_old = current_centroids+0
                #print(centroids_old.shape)
                sse, nearest_indices = self._get_sse_and_nearest(X,current_centroids)
                current_centroids = self._update_centroids(X, nearest_indices, n_clusters)

                if tol>0:
                    # stop if center shift is bewlow tolerance
                    center_shift_total = self.squared_norm(centroids_old - current_centroids)
                    if center_shift_total < tol:
                        if self.verbose > 0:
                            print ("tolerance reached:", center_shift_total.numpy(), " < ", tol.numpy())
                        break

            # get most current sse (has possibly changed through last centroid update)
            sse = self._get_sse(X,current_centroids)                        
            if sse_min is None or sse < sse_min:
                # memorize best solution so far
                sse_min = sse
                best_centroids = current_centroids
                best_n_iter = i+1


            if not isinstance(self.init, str):
                # ndarray as init, no other trials needed
                break
        return best_centroids, sse_min, best_n_iter, init_duration


    def _get_sse_and_nearest(self, samples, centroids):
        """compute sse and - for each data point - the nearest centroid"""

        if self.TF2:
            # shape is int vector in tf 2.0
            n = samples.shape[0]
            d = samples.shape[1]
            k = centroids.shape[0]
        else:
            # shape is Dimension object in tf 1.x
            n = samples.shape[0].value
            d = samples.shape[1].value
            k = centroids.shape[0].value       
        matrix_size = n*d*k*4 # float is 4 byte
        ready = False
        while not ready:
            if matrix_size <= self.max_mem:
                #
                # calculate with full matrices (for smaller data sets)
                #
                self.fract_ = False
                expanded_vectors = tf.expand_dims(samples, 0) #[1,n,d]
                expanded_centroids = tf.expand_dims(centroids, 1) # [k,1,d]


                try:
                    # this internally creates matrix [k,n,d] of float32
                    # e.g. k=100 n=1000000 d=10 ==> 10**9*4 bytes = 4GB                   
                    distances = tf.reduce_sum( tf.square(
                            tf.subtract(expanded_vectors, expanded_centroids)), 2) #[k,n]
                    ready = True
                except Exception as e:
                    # allocation of matrix_size bytes failed
                    print("matrix too large: bytes ",matrix_size,e)
                    self.max_mem = matrix_size-1
                    print("max_mem is now:", self.max_mem, " (use the largest working value for the KMeansTF constructor)")
                if ready:
                    sse = tf.reduce_sum(tf.reduce_min(distances,0)).numpy()
                    nearest = tf.argmin(distances, 0) #[n]
            else:
                #
                # partition the data set into multiple subsets of equal size 
                # (last one is possibly smaller)
                # this is for data sets so large that the required matrices do not
                # fit in GPU memory (adapted to GTX 1060 6MB only, sorry :-))
                # for other cards it my be useful to set self.max_mem differently
                #
                self.fract_ = True
                fractions = math.ceil(1.0*matrix_size/self.max_mem)
                sse = 0
                nearest = None
                # loop of partitions of data set
                for f in range(fractions):
                    b1=f*n//fractions
                    b2=(f+1)*n//fractions
                    if f == fractions-1:
                        # last partition, take the remaining data
                        b2=n
                    expanded_vectors = tf.expand_dims(samples[b1:b2], 0) #[1,n//fractions,d]
                    expanded_centroids = tf.expand_dims(centroids, 1) # [k,1,d]
                    frac_size=(b2-b1)*d*k*4
                    if self.verbose > 0 and f==0:
                        print("frac_size=",frac_size)
                    # this internally creates matrix [k,n//fractions,d] of float32
                    try:
                        distances_f = tf.reduce_sum( tf.square(
                                tf.subtract(expanded_vectors, expanded_centroids)), 2) #[k,n//fractions]
                        self.verbose > 0 and print("handling fraction",f,"of",fractions)
                    except Exception as e:
                        # allocation of frac_size bytes failed
                        print("matrix is too large: bytes: {:,}".format(frac_size))#,e)
                        self.max_mem = int(frac_size*0.9)
                        print("max_mem is now:", self.max_mem, " (use the largest working value for the KMeansTF constructor)")
                        break
                    # compute partial SSE
                    sse_f = tf.reduce_sum(tf.reduce_min(distances_f,0)).numpy()
                    sse += sse_f
                    # compute partial list of nearest
                    nearest_f = tf.argmin(distances_f, 0) #[n]
                    if nearest is None:
                        nearest = nearest_f
                    else:
                        nearest = tf.concat([nearest,nearest_f],axis=0)
                else:
                    # loop did end normally
                    ready = True

        return sse, nearest

    def _get_sse(self, samples, centroids):
        """get summed squared error for given samples and centroids"""
        sse,_ = self._get_sse_and_nearest(samples,centroids)
        return sse

    def _get_nearest(self, samples, centroids):
        """get nearest centroid for each data point"""
        _,nearest = self._get_sse_and_nearest(samples,centroids)
        return nearest

    def _update_centroids(self, samples, nearest_indices, n_clusters):
        """compute new centroids as the mean of all samples associated with a centroid.
        nearest_indices: indicates for each data set the number of the closest centroid"""
        nearest_indices = tf.cast(nearest_indices, tf.int32)
        # determine for each centroid the set of signals for which it is nearest
        partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
        # move each centroid to center of gravity of associated set
        new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
        return new_centroids