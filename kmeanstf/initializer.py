import tensorflow as tf
import numpy as np

import math
import random
import time
import sys
if tf.__version__[0]=="1":
    # for tensorflow 1.4 we need to activate eager execution (for 2.x this is automatic)
    tf.compat.v1.enable_eager_execution()

class MyProblem(Exception):
    pass

class Initializer:
    """provides several methods to initialize a codebook
       most importantly k-means++
    """
    @classmethod
    def init(self,data,k,method="k-means++"):
        """ general wrapper methods for all initialisation methods
            (there were more at some point, thus the class :-))
        """

        if tf.is_tensor(data):
            self.data = data
        else:
            self.data=tf.constant(self.data)
             
        assert(self.data.shape[0]>0)
        assert(k>0)
        self.k = k
        # plausibility check of k
        if k > self.data.shape[0]:
            raise MyProblem("k too large:"+str(k)+" for datasize:"+str(self.data.shape[0]))
        else:
            self.k = k
        if method == "random":
            res = self.initRandom()
        elif method == "k-means++":
            res = self.initKMppTF()  
        else:
            raise MyProblem("wrong init value:"+method)

        return res  

    @classmethod
    def initRandom(self):
        """initialize with random elements
        """
        _tmp=tf.random.shuffle(self.data)
        return _tmp[:self.k],None

    @classmethod
    def initKMppTF(self):
        """k-means++ initialization, ported from scipy implementation
        to use TENSORFLOW
        
        data set is in self-data"""
        if not tf.is_tensor(self.data):
            # self.data is no tensor, convert to tf tensor
            self.data=tf.constant(self.data)

        n_samples, n_features = self.data.shape
        n_clusters = self.k
        # number of local trials
        n_local_trials = 2 + int(np.log(n_clusters))# taken from scikit

        # initialize codebook with zeros
        centers=np.zeros((n_clusters, n_features), dtype=np.float32)
        # choose initial center
        center_id = random.randint(0,n_samples-1) 
        # store in in codebook
        centers[0]=self.data[center_id].numpy()
        
        # compute squared Euclidean distance between initial center and given data set
        closest_dist_sq=tf.square(tf.norm(self.data-tf.constant(centers[0],dtype=tf.float32), axis=1))
        # SSE
        current_pot = tf.reduce_sum(closest_dist_sq)
        # Pick the remaining n_clusters-1 points
        for c in range(1,n_clusters):
            # determine n_local_trials random values in the range [0,current_pot]
            rand_vals=tf.constant(np.random.random(n_local_trials).astype(float)*current_pot,dtype=tf.float32)
            
            # compute positions (0 <= p <= n_samples) where random values would fit in cumulated sum series
            candidate_ids = tf.searchsorted(tf.cumsum(closest_dist_sq), rand_vals)
            # prevent index error since index can be n (too large)
            candidate_ids = tf.minimum(candidate_ids, n_samples-1)
            # decide which candidate is the best, i.e. leads to the lowest overall error/potential
            best_candidate = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                # compute squared Euclidean distance between current candidate and given data set
                dis_curcand=tf.square(
                    tf.norm(
                        self.data-self.data[candidate_ids[trial]]
                        ,axis=1)
                    )
                # determine minimum distance for all data points considering the new candidate
                new_dist_sq=tf.minimum(closest_dist_sq,dis_curcand)
                # compute new potential
                new_pot=tf.reduce_sum(new_dist_sq)
                # Store result if it is the best local trial so far
                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            # centers contains centroids in order of placement!
            centers[c]=self.data[best_candidate] 
            current_pot = best_pot
            closest_dist_sq=best_dist_sq
        return tf.convert_to_tensor(centers, np.float32), best_pot

