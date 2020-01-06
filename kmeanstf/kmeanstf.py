"""tensorflow-based implementation of algorithms for the k-means problem

* k-means (class KMeansTF with init="random")
* k-means++ (class KMeansTF with init="k-means++")
* tunnel k-means (class TunnelKMeansTF)



"""

# Author: Bernd Fritzke <fritzke@web.de>
# License: BSD 3 Clause
import tensorflow as tf
import nvidia_smi
import numpy as np
import math
import random
import os
import sys

from time import time
from sklearn.cluster import KMeans
from ._initializer import _Initializer
import itertools
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# TF_CPP loglevel: 
# 0: all messages
# 1: INFO filtered out
# 2: WARNING filtered out
# 3: ERROR filtered out
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"]="" # NOSONAR


class _MyProblem(Exception):
    pass

class InsufficientGpuMemoryError(Exception):
    # raised when GPU is available but has to little free memory
    pass

class TensorError(Exception):
    # raised when expected tensor-variable is no tensor
    pass

class InvalidInitTypeError(Exception):
    # raised when init parameter is of unexpected type
    pass


class BaseKMeansTF:
    """
Base class for :class:`.KMeansTF` and :class:`.TunnelKMeansTF`

.. note::

    Recommended usage of this class is via the derived classes :class:`.KMeansTF` and :class:`.TunnelKMeansTF`
    To use :class:`.BaseKMeansTF` directly, set the parameter ``tunnel`` to False (k-means/k-means++) or True (tunnel k-means).

Args:
    n_clusters (int): The number of clusters to form as well as the number of centroids to generate.
    init ('random', 'k-means++' or array): method of initialization
    n_init (int): number of runs of the initial k-means phase with different initializations (default 1). 
                    Only one tunnel phase is performed even if n_init is larger than 1.
    max_iter (int):   Maximum number of Lloyd iterations for a single run of the k-means algorithm.
    tol (float): Relative tolerance with regards to inertia to declare convergence.
    verbose (int):    Verbosity mode.
    random_state (int): None, or integer to seed the random number generators of python, numpy and tensorflow
    tunnel (boolean): perform tunnel k-means?
    max_tunnel_iter (int): how many tunnel iterations to perform maximally
    max_tunnel_moves_per_iter (int): how many centroids to move maximally in one tunnel iteration
    criterion (float): inital required ratio error/utility (is increased adaptively)
    local_trials (int): how many time should each tunnel move be repeated with different random offset vector (1 or larger)
    collect_history (bool): collect historic information on inertia, criterion, tunnel moves, codebooks

Attributes:
    cluster_centers_ (array, [n_clusters, n_features]): Coordinates of cluster centers. If the algorithm stops before fully 
        converging (see tol and max_iter), these will not be consistent with labels\\_.
    labels_ (array, shape(n_samples)): Labels of each point, i.e. index of closest centroid
    inertia_ (float): Sum of squared distances of samples to their closest cluster center.
    n_iter_ (int): Number of iterations run.


.. autosummary::

    """
    _num_type = np.float32
    _num_type_tf=tf.float32

    @classmethod
    def _use_32(cls):
        """use 32 bit datatypes for all computations"""
        BaseKMeansTF._num_type = np.float32
        BaseKMeansTF._num_type_tf = tf.float32
        _Initializer._use_32()

    @classmethod
    def _use_64(cls):
        """use 64 bit datatypes for all computations"""
        BaseKMeansTF._num_type = np.float64
        BaseKMeansTF._num_type_tf = tf.float64
        _Initializer._use_64()
   
    @staticmethod
    def _get_num_type():
        return BaseKMeansTF._num_type

    @staticmethod
    def _get_num_type_tf():
        return BaseKMeansTF._num_type_tf

    def __init__(self,  # NOSONAR
                 n_clusters=8,
                 init='k-means++',
                 n_init=10,
                 max_iter=300,
                 tol=1e-4,
                 # from here
                 verbose=0,
                 random_state=None,
                 tunnel=False,
                 max_tunnel_iter=300,
                 max_tunnel_moves_per_iter=100,
                 criterion=1.0,
                 local_trials=1,
                 collect_history=False):
        self._system_status = KMeansTF.get_system_status()
        self._good_max_mem = self._get_good_max_mem()
                # from scikit learn
        self.n_clusters = n_clusters
        # convert numpy codebook to tf
        if isinstance(init,str):
            self.init = init
            self.cluster_centers_= None
        else:
            self.init = self._ensure_tensor(init)
            self.cluster_centers_ = self.init+0

        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        TF2 = tf.__version__[0] == "2"
        if not self.random_state is None:
            if TF2:
                tf.random.set_seed(self.random_state)
            else:
                tf.compat.v1.set_random_seed(self.random_state)
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        self.n_init_ = n_init
        #custom parameters
        self.tunnel_ = tunnel
        self.max_tunnel_iter = max_tunnel_iter
        self.max_tunnel_moves_per_iter = max_tunnel_moves_per_iter
        self.criterion_ = criterion  # start value for criterion
        self.local_trials_ = local_trials
        self.collect_history = collect_history
        self.criterion_factor_ = 1.1
        # relative to mean distance beween neighboring centroids
        self.source_freeze_factor = 1.1  # 3.0 # freeze in the vicinity of source
        self.target_freeze_factor = 3.0  # freeze in the vicinity of target
        #
        # self.max_men is an empirically found number to indicate when the 3D-matrix used for distance computation
        # should be partitioned to avoid memory overflow of the GPU
        # the default value is empirically found for NVidia GTX-1060 6MB
        # may need to be adapted for other GPUs
        # if memory allocation errors occur, a sequence of smaller and smaller max_mem value is tried
        # and printed until the computation is successfull.
        # This makes it possible to find the best value for the GPU in use
        # successfully used value for colab GPU: 4000000000
        #self.max_mem = max_mem
        # must work with tensorflow 1.4 and tensorflow 2.0b
        # in some places different code is needed

        # noinspection PyUnresolvedReferences
        # pylint: disable=maybe-no-member
        # self.TF2 = tf.__version__[0] == "2"


        #print(f"good maxmem: {self._good_max_mem:,d}")

        self.max_mem = self._good_max_mem + 0
        # result variable
        self.inertia_ = None
        #
        # class-global variables, not always set
        #
        self.tunnel_history = []  # history of tunnel moves
        self.criterion_history = []
        self.codebook_history = []
        self.error_history = []
        self.utils_ = None  # utilities of current codebook
        self.errs_ = None  # error of current codebook
        self.moves_ = None  # list of [source,target] pairs for moving
        self.factors_ = None  # list of factors valid for moved units
        self.frozen_ = None  # tensor of booleans to indicate frozen centroids
        self.to_move_ = None  # tensor of booleans to indicate centroids to move
        self.no_of_lloyd = 0  # number of Lloyd iterations performed
        #
        # parameters given to this instance
        #
        self._params = {
            'n_clusters': n_clusters,
            'init': init,
            'n_init': n_init,
            'max_iter': max_iter,
            'tol': tol,
            'max_mem': self.max_mem,
            'tunnel': tunnel,
            'max_tunnel_iter': max_tunnel_iter,
            'max_tunnel_moves_per_iter': max_tunnel_moves_per_iter,
            'criterion': criterion,
            'local_trials': local_trials
        }
        if isinstance(init, tf.Tensor):
            shape_str = str(KMeansTF._shape(init))
            self._params['init'] = 'array of shape '+shape_str
 
        #
        # results of simulation run
        #
        self._log = {
            #'start_time': None,
            # time used for initialization (random or k-means++ init)
            'model':type(self).__name__,
            'data': None,
            'k':self.n_clusters,
            'init_duration': None,
            'kmeans_inertia': None,  # after initialization, e.g. kmpp+Lloyd
            'lloyd_duration': None,  # init_duration + time for Lloyd iters
            'kmeans_lloyd_iter': None,  # Lloyd iteration in init phase
            'tunnel_inertia': None,  # after tunnel phase
            'tunnel_lloyd_iter': None,  # Lloyd iteration in tunnel phase
            'tunnel_duration': None,  # duration of tunnel phase
            'collect_history': self.collect_history,  # set to True to collect
            'final_inertia': None,  # result
            'total_duration': None,  # overall time
        }

        #
        # history items with lengthy entries are handled here
        #
        self._history = {
            'criterion_history': None,
            'error_history': None,
            'tunnel_history': None,
            # implicitely contains
            # number of tunnel iterations
            # number of tunnel moves
            'codebook_history': None
        }

    _log_abbr = {
        #'start_time': 'stat',
        'init_duration': 'indu',
        'kmeans_inertia': 'kmin',
        'lloyd_duration': 'lldu',
        'kmeans_lloyd_iter': 'klit',
        'tunnel_inertia': 'tuin',
        'tunnel_lloyd_iter': 'tlit',
        'tunnel_duration': 'tudu',
        'collect_history': 'cohi',
        'final_inertia': 'fiin',
        'total_duration': 'todu'
        }

    def get_params(self):
        """Get params used to define class
        
        Returns:
            params (dict)"""
        return self._params

    def get_log(self, abbr = False):
        """Get statistics of performed run of fit()
        
        Args:
            abbr (bool): return with abbreviated keys
            
        Returns:
            log (dict)"""
        if abbr:
            ret = {}
            for k in self._log:
                if k in self.__class__._log_abbr:
                    ret[self.__class__._log_abbr[k]]=self._log[k]
                else:
                    ret[k]=self._log[k]
            return ret
        else:
            return self._log

    def get_history(self):
        """Get collected history data of performed run of fit().\n
        (only present if collect_history == True)
        
        Returns:
            history (dict)"""
        return self._history

    def _uses_gpu(self):
        return self._system_status["tf_logi_gpus"] != 0
    
    def _get_good_max_mem(self):
        if self._uses_gpu():
            tot_mem = self._system_status["gpu_mem_total"]
            gtxmm = 2147483648 # gtx 1080 max_mem, 2GB
            gtxtm = 6365118464 # gtx 1080 total mem
            max_mem = max(int(tot_mem/(gtxtm/gtxmm)),tot_mem - (gtxtm-gtxmm))
        else:
            max_mem = 1_000_000_000_000 # infinity ...
        return max_mem

    def _perform_tunnel_phase(self, X):  # NOSONAR
        """tunnel phase: 
           * perform one or more tunnel iterations
           * each tunnel iteration consists of a number of tunnel moves
           * each tunnel move relocates a low-utility centroid towards a high-error centroid
        """

        prev_cb = self.cluster_centers_+0
        prev_inertia = self.inertia_
        e1 = self.inertia_
        crit = self.criterion_

        #
        # perform tunnel iterations up to max number
        #
        tunnel_lloyd_iter = 0
        for i in range(self.max_tunnel_iter):
            if self.verbose >= 1 and i == 0:
                print(
                    "\n_perform_tunnel_phase(): starting error: {:5.5f}".format(e1))

            # plan moves
            self.moves_, self.factors_, self.frozen_, self.to_move_ = \
                self._plan_moves(X, self.cluster_centers_, criterion=crit)

            # execute moves
            if len(self.moves_) == 0:
                # no moves ...
                if self.verbose >= 1:
                    print(f"tunneliter #{i:d} moves:  0, break!")
                if (self.inertia_ > prev_inertia):
                    # revert to best-so-far solution
                    self.cluster_centers_ = prev_cb
                    self.inertia_ = prev_inertia
                if self.collect_history:
                    # save stuff for history
                    self.tunnel_history.append([])
                    self.criterion_history.append(crit)
                    self.codebook_history.append(self.cluster_centers_.numpy())
                    self.error_history.append(self.inertia_)

                if self.verbose >= 1:
                    print("final SSE: {:5.5f}   improvement: {:5.5%}".format(
                        self.inertia_,  (e1-self.inertia_)/e1))
                break
            else:
                # some moves to perform!

                if self.collect_history:
                    # save stuff for history
                    self.tunnel_history.append(self.moves_)
                    self.criterion_history.append(crit)
                    self.codebook_history.append(self.cluster_centers_.numpy())
                    self.error_history.append(self.inertia_)

                # store old centers
                old_centers = self.cluster_centers_ + 0
                besterr = None
                bestcb = None
                #
                # additional search by repeating the (randomized) moves
                #
                for _ in range(self.local_trials_):
                    self.cluster_centers_ = old_centers+0
                    #
                    # perform the moves!
                    #
                    self._perform_moves(self.moves_)

                    # we run k-means on the now current cluster centers (resulting from the moves)!
                    self.init = self.cluster_centers_
                    tmp = self.verbose
                    self.verbose = 0
                    #
                    # k-means!
                    #
                    self.cluster_centers_, self.inertia_, self.n_iter_, _, _, total_n_iter = \
                        self._k_means(
                            X,
                            n_clusters=self.n_clusters,
                            init=self.cluster_centers_, # no random or k-means++
                            n_init=1, # only one run. since init is codebook
                            max_iter=self.max_iter,
                            tol=self.tol,
                            direct_call_from_fit=False)
                    tunnel_lloyd_iter += total_n_iter

                    self.verbose = tmp
                    e2 = self.inertia_
                    if besterr == None or e2 < besterr:
                        besterr = e2
                        bestcb = self.cluster_centers_+0

                self.inertia_ = besterr
                self.cluster_centers_ = bestcb

                if self.verbose >= 1:
                    print("tunneliter #{:d} moves: {:2d}  SSE: {:5.5f}  improvement: {:5.5%} crit={:5.5f}".format(
                        i, len(self.moves_), e2, (e1-e2)/e1, crit), end="   ")
                if self.verbose >= 2:
                    print()

                if e2 >= prev_inertia:
                    # solution deteriorated:
                    # ==> go back to previous solution and sharpen criterion
                    self.cluster_centers_ = prev_cb
                    self.inertia_ = prev_inertia
                    # minimum criterion increase
                    crit *= self.criterion_factor_
                    # ensure that increase is large enough to prevent the last jump
                    if crit < self.factors_[-1]:
                        if self.verbose >= 1:
                            print("CRIT JUMP", end="   ")
                        #
                        # change criterion such that most recent jump is not possible anymore!
                        # (not always the same vanilla change)
                        crit = self.factors_[-1]*1.01
                    if self.verbose >= 1:
                        print(
                            "SSE deteriorated, raising criterion to {:5.5f}".format(crit))
                else:
                    # solution improved, take it as best-so-far and continue
                    prev_cb = self.cluster_centers_+0
                    prev_inertia = e2
                    if self.verbose >= 1:
                        print()
        else:
            if self.verbose >= 1:
                print("max no of tunnel iters reached: ", i+1)
        self._log["tunnel_lloyd_iter"] = tunnel_lloyd_iter
        self._history["tunnel_history"] = self.tunnel_history
        self._history["criterion_history"] = self.criterion_history
        self._history["error_history"] = self.error_history
        self._history["codebook_history"] = self.codebook_history
        self.crit_ = crit

    def _tolerance_skl(self, X, tol):
        """Return a tolerance which is independent of the data set"""
        variances = np.var(X.numpy(), axis=0)
        return np.mean(variances) * tol

    def _tolerance(self, X, tol):
        """Return a tolerance which is independent of the data set"""
        _, variances = tf.nn.moments(X, axes=[0])
        return tf.math.reduce_mean(variances) * tol

    def _multi_fit(self, X, n):
        for i in range(n):
            self.fit(X)
            print(i, self.inertia_)

    def _ensure_tensor(self, X):
        # convert X to Tensor if needed
        # convert to correct num_type if needed
        if isinstance(X, tf.Tensor):
            if not X.dtype == self._get_num_type_tf():
                # cast to required type
                X = tf.cast(X,self._get_num_type_tf())
        elif isinstance(X, np.ndarray):
            # create tf.Tensor from to ndarray
            X = tf.constant(X.astype(self._get_num_type()))
        else:
            raise TensorError(
                "Type for parameter X must be ndarray or tf.Tensor but is "+str(type(X)))
        return X
       
    def fit(self, X):
        """Compute k-means clustering.

        Args:
            X (tensor): samples

        sets:

            * self.cluster_centers\\_
            * self.inertia\\_
        """

        X = self._ensure_tensor(X)
        t_start = time()

        #
        # perform standard k-means (with e.g. random init or k-means++)
        #
        self.cluster_centers_, self.inertia_, self.n_iter_, \
            self._log["init_duration"], self._log["lloyd_duration"], \
            self._log["kmeans_lloyd_iter"],     = \
            self._k_means(
                X, 
                n_clusters=self.n_clusters,
                init=self.init,
                n_init=self.n_init_,
                max_iter=self.max_iter,
                tol=self.tol, 
                direct_call_from_fit=True)
        self._log["kmeans_inertia"] = self.inertia_

        #
        # perform tunnel k-means (optional)
        #
        if self.tunnel_:
            #self.tunnel_ = False  # only tunnel at the first run
            t_start_tunnel = time()
            # calls _k_means() again to do k-means on given codebook
            self._perform_tunnel_phase(X)
            self._log["tunnel_inertia"] = self.inertia_
            self._log["tunnel_duration"] = time()-t_start_tunnel

        self._log["total_duration"] = time()-t_start
        self._log["final_inertia"] = self.inertia_
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Args:
            X (tensor): samples

        Returns:
            array of cluster indices
        """
        X = self._ensure_tensor(X)
        return self._get_nearest_centroids(X, self.cluster_centers_)

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample.

        Args:

            X (tensor): samples

        Returns:

            array of cluster indices

        """
        X = self._ensure_tensor(X)
        self.fit(X)
        return self.predict(X)

    def _squared_norm(self, X):
        """compute squared norm of given tensor X"""
        return tf.reduce_sum(tf.square(tf.norm(X)))

    def _assert_not_nan(self, x, loc="nowhere"):
        aaa = x.numpy().flatten()
        if np.any(np.isnan(aaa)):
            print("NAN detected:", loc, "#:",np.sum(np.isnan(aaa)), " of ",len(aaa))

    def _k_means(self, X, n_clusters, init='k-means++',  # NOSONAR
                 n_init=10, max_iter=300,
                 tol=1e-4, direct_call_from_fit=False):
        """K-means clustering algorithm

        possible initializations:

        - "random"
        - "k-means++"
        - actual codebook (detected as not being a string)

        can be called from fit() or if called from tunnelling
        * call from fit(): several possibilities for init(random, k-means++,codebook)
        * call from tunnelling: always init=codebook

        """
        # initialize
        sse_min = None
        best_n_iter = 0
        total_n_iter = 0
        # compute X-dependent tolerance value
        tol = tf.cast(self._tolerance(X, tol),self._get_num_type_tf())
        #
        # repeat several times?
        #
        if isinstance(init, tf.Tensor):
            # init is codebook, just run once in this case
            no_of_runs = 1
        else:
            # random or k-means++, do n_init trials
            no_of_runs = self.n_init_

        #
        # run k-means no_of_runs times with different initializations
        #
        init_duration = 0.0
        lloyd_duration = 0.0
        for n_i in range(no_of_runs):  # n_init runs of k-means
            #
            # initialization (random, k-means++ or codebook)
            #
            if self.verbose >= 1:
                print(
                    f"\n_k_means(): k-means init #{n_i}, init={init}", end="  ")
            if self.verbose >= 2:
                print()
            start = time()
            if isinstance(self.init, str):
                # init indicates a method, only happens when called from fit()
                if self.init == "random":
                    # random init
                    current_centroids, _ = _Initializer.init(
                        X, n_clusters, "random")
                    #self._assert_not_nan(current_centroids, "init random")
                elif self.init == "k-means++":
                    # k-means++ init
                    current_centroids, _ = _Initializer.init(X, n_clusters, "k-means++")
                    #self._assert_not_nan(current_centroids, "init k-means++")
                else:
                    raise _MyProblem("not valid as init value: "+self.init)
            elif isinstance(self.init, tf.Tensor):
                # init is a codebook
                current_centroids = self.init
                #self._assert_not_nan(current_centroids, "init cb")
            else:
                raise InvalidInitTypeError("invalid type for init:"+str(type(self.init)))
            init_duration += time()-start  # duration of random or k-means++ init or assignment
            if direct_call_from_fit and self.collect_history:
                # remember codebook (overwritten if several runs, i.e. n_init>1)
                full_cb_history = [current_centroids+0]
            init_codebook = current_centroids+0
            #
            # perform Lloyd Iterations until convergence
            #
            start_lloyd = time()
            for i in range(max_iter):  # maybe zero, if we just want initialization
                if self.verbose >= 2:
                    print(
                        f"Lloyd #{i:2} sse={sse:10.7f} tol={tol.numpy():.5E}")
                # create copy to measure tol
                centroids_old = current_centroids+0
                sse, nearest_indices = self._get_sse_and_nearest_centroids(X, current_centroids, direct_call_from_fit)
                #
                # sometimes nan's occur, replace with centroids_old
                # this is not a bug. It can e.g. happen if a centroid is in the middle of
                # two clusters and in the next Lloyd iteration boundaries shift such that
                # it is left without any data points
                # Example (1D)
                # ............
                # ***   *X   *     Data Points (X stands for e.g. 100 samples in one place, * is a single sample)
                #  ^^        ^     3 centroids
                # (now perform 1 Lloyd iteration)
                # 
                # ***   *X   *        
                #  ^  ^  ^         middle centroid is dead now
                # 
                current_centroids = tf.cast(self._update_centroids(
                    X, nearest_indices, n_clusters), self._get_num_type_tf())
                if direct_call_from_fit and self.collect_history:
                    # remember codebook
                    full_cb_history.append(current_centroids+0)
                if tf.math.reduce_any(tf.math.is_nan(current_centroids)):
                    # nan's present
                    #print("NAN found!")
                    self.nancenters = centroids_old+0
                    # we need to assign a valid value to all codebook vectors where
                    # tf.math.is_nan(current_centroids) is true
                    current_centroids = tf.where(tf.math.is_nan(current_centroids), tf.gather(
                        X, self.nearest_samples), current_centroids)

                nearest_indices = None  # free
                self.no_of_lloyd += 1

                if tol > 0:
                    # stop if center shift is below tolerance
                    center_shift_total = tf.cast(self._squared_norm(
                        centroids_old - current_centroids),self._get_num_type_tf())

                    if center_shift_total < tol:
                        if self.verbose > 0:
                            print(
                                f"tolerance reached at Lloyd iter {i:d} SSE={sse:5.5f}: {center_shift_total.numpy():.5E} < {tol.numpy():.5E} (tol)")
                        break
                    else:
                        if self.verbose > 10:
                            print("center_shift_total=", center_shift_total)
            if max_iter > 0:
                total_n_iter += i+1  # counts all Lloyd iterations in this call of _k_means
                lloyd_duration += time() - start_lloyd
            else:
                # no Lloyd iterations
                lloyd_duration = None
                total_n_iter = None

            # get most current sse (has possibly changed through last centroid update)
            sse = self._get_sse(X, current_centroids)
            if self.verbose >= 1:
                print(" k-means SSE: {:5.5f}".format(sse))
            if ((sse_min is None) or (sse < sse_min)):
                # this is the best solution so far
                # memorize it
                sse_min = sse
                best_centroids = current_centroids
                if (max_iter > 0):
                    # did we perform any Lloyd iters
                    best_n_iter = i+1
                else:
                    best_n_iter = 0
                self.best_init_codebook = init_codebook+0
                # and remember init codebook as well

        if direct_call_from_fit and self.collect_history:
            # remember codebooks
            self.full_cb_history = full_cb_history
        return best_centroids, \
            sse_min, \
            best_n_iter, \
            init_duration,  \
            lloyd_duration, \
            total_n_iter

    def _rand_unit_vec(self, d):
        """ random vector from surface of d-dimensional unit hypersphere
        """
        inv_d = 1.0 / d
        gauss = np.random.normal(size=d)
        length = np.linalg.norm(gauss)
        if length == 0.0:
            x = gauss
        else:
            r = np.random.rand() ** inv_d
            x = np.multiply(gauss, r / length)
        length = np.linalg.norm(x)
        x /= length
        return(x)

    def _perform_moves(self, moves):
        """ perform a sequence of moves among the clustercenters

            moves: list of [source, target] pairs,
                      e.g.: [[15, 40], [39, 41], [13, 16]]

        """
        # perform planned moves (in numpy, since tf constants are not mutable)
        if self.verbose >= 2:
            print("_perform_moves(), namely ", len(moves))
        tmp = self.cluster_centers_.numpy()
        d = int(KMeansTF._shape(self.cluster_centers_)[1])

        for source, target in moves:
            # add a tiny offset vector so that both get different Voronoi regions
            # offset is made dependent on means cluster distance to work with
            # data sets in any range
            offset = self._rand_unit_vec(d)*0.0001*self.mean_c_dist_
            tmp[source] = self.cluster_centers_[target].numpy()+offset
        self.cluster_centers_ = tf.constant(tmp)

    def get_errs_and_utils(self, X, centroids=None):
        """Get error and utility values wrt. X
        
        Args:
            X (tensor): samples

        Error and utility are computed for given centroids 
        or (if centroids = None) for self.cluster_centers\\_
        
        Returns:
            errors (array), utilities (array)"""
        if centroids is None:
            """assume current cluster_centers"""
            centroids = self.cluster_centers_
        self._plan_moves(X, centroids, 1000)
        return self.errs_, self.utils_
    def _numsize(self):
        return self._get_num_type()(1).itemsize
    def _plan_moves(self, X, centroids, criterion):  # NOSONAR
        """
        plan a tunnel iteration

        * compute sse and utility for each centroid
        * prepare list of moves of sufficiently useless units towards the most error-loaden ones

        side-effects:
        sets self._utils, self._errs
        """
        n_clusters = self.n_clusters
        self.cluster_centers_

        # shape is int vector in tf 2.0 but Dimension in tf1.4
        n = KMeansTF._shape(X)[0]
        d = KMeansTF._shape(X)[1]
        k = KMeansTF._shape(centroids)[0]

        matrix_size = n*d*k*self._numsize()  # size of matrix used for distance computation float is e.g. 4 byte
        ready = False
        while not ready:
            if matrix_size <= self.max_mem:
                #
                # try to calculate with full matrices (for smaller data sets)
                #
                self.fract_ = False  # no splitting of the data set
                expanded_vectors = tf.cast(tf.expand_dims(X, 0),self.__class__._num_type_tf)  # [1,n,d]
                expanded_centroids = tf.cast(tf.expand_dims(centroids, 1),self.__class__._num_type_tf)  # [k,1,d]
                try:
                    # this internally creates matrix [k,n,d] of self._get_num_type()
                    # e.g. k=100 n=1000000 d=10 ==> 10**9*4 bytes = 4GB
                    #
                    # distances is kxn matrix containing the distance of
                    # each (1 of n) data point to each (1 of k) cluster center
                    #
                    distances = tf.reduce_sum(tf.square(
                        tf.subtract(expanded_vectors, expanded_centroids)), 2)  # [k,n]
                    # no SSE computation here ....
                    # no nearest centrid computation here
                    expanded_vectors = None  # free
                    expanded_centroids = None  # free
                    ready = True
                except Exception as e:
                    #
                    # matrix too large to fit into memory
                    #
                    expanded_vectors = None  # free
                    expanded_centroids = None  # free

                    # sys.exc_info(): the values returned are (type, value, traceback)
                    exc_type, _, exc_tb = sys.exc_info()  # (type, value, traceback)
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    if self.verbose >= 1:
                        print("Exception during distance computation:", exc_type, fname, exc_tb.tb_lineno)
                    # allocation of matrix_size bytes failed
                    print(f"plan_moves(): matrix too large for GPU: bytes {matrix_size:,d}, k={k} n={n:,} d={d} numsize={self._numsize()}, max_mem={self.max_mem:,}")
                    print("automated slicing started")
                    if self.verbose < 1:
                        print("set 'verbose' to 1 to see more details")
                    if self.verbose >= 1:
                        print(e)                    
                    # print("plan_moves(): matrix 0 too large: bytes {:,d}".format(
                    #     matrix_size), e)
                    # self.max_mem = matrix_size-1  # sufficient to achieve a quite smaller size next time
                    # if self.verbose >= 1:
                    #     print("max_mem is now:", self.max_mem,
                    #           " (use the largest working value for the KMeansTF constructor)")
                    continue  # with new self.max_mem value

                # assertion: distances matrix was successfully computed

                try:
                    #  compute indices of sorted distances
                    bmu_ind = tf.argsort(distances, axis=0)[0]  # [n]
                except Exception as e:
                    ready = False
                    # allocation of matrix_size bytes failed
                    if self.verbose >= 1:
                        print("plan_moves(): too large to argsort ",
                              distances.shape)
                        print("ERROR:", e)
                    self.max_mem = matrix_size-1  # sufficient to achieve a quite smaller size next time
                    if self.verbose >= 1:
                        print("plan_moves(): max_mem is now:", self.max_mem,
                            " (use the largest working value for the KMeansTF constructor)")
                    distances = None  # free
                    continue  # with new self.max_mem value

                # compute distances to bmu and bmu2 for all samples
                speedy = True
                if speedy:
                    # find smallest 2 distances for each sample
                    distances = tf.transpose(-distances)
                    m, _ = tf.math.top_k(distances, 2)
                    sort_dist = tf.transpose(-m)
                else:
                    sort_dist = tf.sort(distances, axis=0)[:2]  # [2,n]
                distances = None  # free
                # bmu-distances are in sort_dist[0], bmu2-distances are in sort_dist[1]

                # compute all per-sample utilities (for each sample, how useful is its bmu?)
                utility = sort_dist[1]-sort_dist[0]  # [n]
                # compute all per-sample errors (for each sample, how distant is its bmu?)
                error = sort_dist[0]  # [n]
                sort_dist = None  # free

                # determine for each centroid the utilities of its Voronoi set
                u_partitions = tf.dynamic_partition(
                    utility, bmu_ind, n_clusters)
                # compute per-centroid utilities [k]
                self.utils_ = tf.concat(
                    [tf.expand_dims(tf.reduce_sum(u_partition, axis=0), 0)
                     for u_partition in u_partitions], 0)

                # determine for each centroid the errors of its Voronoi set
                e_partitions = tf.dynamic_partition(error, bmu_ind, n_clusters)
                bmu_ind = None  # free
                # compute per-centroid errors [k]
                self.errs_ = tf.concat(
                    [tf.expand_dims(tf.reduce_sum(e_partition, axis=0), 0)
                     for e_partition in e_partitions], 0)
            else:
                # partition the data set into multiple subsets of equal size
                # (last one is possibly smaller)
                # this is for data sets so large that the required matrices do not
                # fit in GPU memory (adapted to GTX 1060 6MB only, sorry :-))
                # for other cards it my be useful to set self.max_mem differently
                #
                self.fract_ = True
                # number of fractions needed such that each is not larger than self.max_mem
                n_frac = math.ceil(1.0*matrix_size/self.max_mem)
                utility = None
                error = None
                bmu_ind = None
                # loop of partitions of data set
                startover = False
                # we divide the sample set into
                for f in range(n_frac):
                    b1 = (f+0)*n//n_frac
                    b2 = (f+1)*n//n_frac
                    if f == n_frac-1:
                        # last partition, take the remaining data
                        b2 = n
                    expanded_vectors = tf.expand_dims(
                        X[b1:b2], 0)  # [1,n//n_frac,d]
                    expanded_centroids = tf.expand_dims(
                        centroids, 1)  # [k,1,d]
                    frac_size = (b2-b1)*d*k*4
                    if self.verbose > 0 and f == 0:
                        print(
                            f"plan_moves(),frac: {n_frac} frac_size={frac_size:,d}")
                    # this internally creates matrix [k,n//n_frac,d] of self._get_num_type()
                    try:
                        if self.verbose > 0:
                            print("plan_moves(), frac: handling fraction",
                                  f, "of", n_frac)

                        distances_f = tf.reduce_sum(tf.square(
                            tf.subtract(expanded_vectors, expanded_centroids)), 2)  # [k,n//n_frac]
                    except Exception as e:
                        # allocation of frac_size bytes failed
                        print("plan_moves(), frac: matrix 0 is too large: bytes: {:,}".format(
                            frac_size))  # ,e)
                        if self.verbose >= 1:
                            print(e)
                        self.max_mem = int(frac_size*0.9)
                        print("plan_moves(), frac: max_mem is now:", self.max_mem,
                              " (use the largest working value for the KMeansTF constructor)")
                        startover = True
                        break

                    try:
                        # indices of distances to bmu
                        bmu_ind_f = tf.argsort(distances_f, axis=0)[
                            0]  # internally: [k,n//n_frac]
                    except Exception as e:
                        # allocation of frac_size bytes failed
                        print("plan_moves(), frac: matrix 2 is too large: bytes: {:,}".format(
                            frac_size))  # ,e)
                        if self.verbose >= 1:
                            print(e)
                        self.max_mem = int(frac_size*0.9)
                        print("plan_moves(), frac: max_mem is now:", self.max_mem,
                              " (use the largest working value for the KMeansTF constructor)")
                        startover = True
                        distances_f = None  # free
                        break

                    # bmu-distances are in sort_dist_f[0], bmu2-distances are in sort_dist_f[1]
                    if self.verbose >= 2:
                        print("plan_moves(), frac: distances_f.shape = ",
                              distances_f.shape)
                    speedy = True
                    if speedy:
                        # find smallest 2 distances for each sample
                        distance_f = tf.transpose((-1)*distances_f)
                        m, _ = tf.math.top_k(distance_f, 2)
                        sort_dist_f = tf.transpose(-m)
                    else:
                        sort_dist_f = tf.sort(
                            distances_f, axis=0)  # [k,n//n_frac]

                    distances_f = None
                    # compute all per-sample utilities
                    utility_f = sort_dist_f[1]-sort_dist_f[0]  # [n//n_frac]
                    # take all per-sample errors
                    error_f = sort_dist_f[0]  # [n//n_frac]
                    sort_dist_f = None  # free

                    # collect bmu_ind from slices
                    if bmu_ind is None:
                        bmu_ind = bmu_ind_f
                    else:
                        bmu_ind = tf.concat([bmu_ind, bmu_ind_f], axis=0)

                    # collect utility from slices
                    if utility is None:
                        utility = utility_f
                    else:
                        utility = tf.concat([utility, utility_f], axis=0)

                    # collect error from slices
                    if error is None:
                        error = error_f
                    else:
                        error = tf.concat([error, error_f], axis=0)
                else:
                    # loop over fractions did end normally
                    ready = True
                if startover:
                    print("plan_moves(): STARTOVER with new max_mem: {:,d}".format(
                        self.max_mem))
                    continue

                # determine for each centroid the utilities of its Voronoi set
                u_partitions = tf.dynamic_partition(
                    utility, bmu_ind, n_clusters)

                # compute per-centroid utilities [k]
                self.utils_ = tf.concat(
                    [tf.expand_dims(tf.reduce_sum(u_partition, axis=0), 0)
                     for u_partition in u_partitions], 0)

                # determine for each centroid the errors of its Voronoi set
                e_partitions = tf.dynamic_partition(error, bmu_ind, n_clusters)
                bmu_ind = None  # free

                # compute per-centroid errors [k]
                self.errs_ = tf.concat(
                    [tf.expand_dims(tf.reduce_sum(e_partition, axis=0), 0)
                     for e_partition in e_partitions], 0)

            #
            # now use all this info to determine the actual moves
            #
            moves, factors, frozen, to_move = self._collect_moves(
                centroids, criterion)
            # side effects!

            return moves, factors, frozen, to_move

    def _collect_moves(self, centroids, criterion):  # NOSONAR
        #
        # based on precomputed utility and error figure out the possible moves
        # depending on the current criterion value
        #

        # compute mutual square distances of centroids
        ex_0 = tf.expand_dims(centroids, 0)  # [1,k,d]
        ex_1 = tf.expand_dims(centroids, 1)  # [k,1,d]

        # square distances
        # internally creates [k,k,d]-matrix
        c_distances = tf.reduce_sum(tf.square(
            tf.subtract(ex_0, ex_1)), 2)  # [k,k]

        # Euclidean distances
        c_distances = tf.sqrt(c_distances)  # [k,k]
        self.c_d = c_distances

        # indices of sorted c-distances
        sort_c_dist_ind = tf.argsort(c_distances, axis=1)  # [k,k]

        # mean distance to nearest other centroid
        self.mean_c_dist_ = tf.reduce_mean(tf.sort(c_distances, axis=1)[:, 1])

        self.source_freeze_radius_ = self.mean_c_dist_ * self.source_freeze_factor
        self.target_freeze_radius_ = self.mean_c_dist_ * self.target_freeze_factor

        # indices of target units (descending by error)
        target_sorted_ind = tf.argsort(self.errs_, direction="DESCENDING")

        # indices of useless units (ascending by utility)
        useless_sorted_ind = tf.argsort(self.utils_)

        # frozen marker (units set to True will not be moved due to a moved neighbor)
        frozen = np.full(self.n_clusters, False)

        # to_move: units set to True are forseen for tunnel moves
        to_move = np.full(self.n_clusters, False)

        # index to the current target unit
        target_cur = 0

        #
        # ACTUALLY PREPARE LIST OF MOVES
        #

        # planned tunneling ops (to be executed later)
        moves = []
        factors = []
        # sequentially loop over units, most useless first
        for source_i in useless_sorted_ind:
            if frozen[source_i]:
                # hands off this unit! (it is frozen)
                continue
            else:
                # assertion: source_i is the index of the unit to move!
                # now determine unfrozen target unit
                while frozen[target_sorted_ind[target_cur]]:
                    target_cur += 1
                    if target_cur == self.n_clusters:
                        break
                if target_cur == self.n_clusters:
                    # no more target units, finish
                    break
                else:
                    # assertion: target_sorted_ind[target_cur] is the index of the unit to support
                    target_i = target_sorted_ind[target_cur]
                    # tunnel condition fulfilled?
                    if self.utils_[source_i]*criterion < self.errs_[target_i]:

                        # plan tunnel move: source ==> target
                        moves.append([source_i.numpy(), target_i.numpy()])
                        to_move[source_i] = True
                        # remember ratio error/utility
                        factors.append(
                            (self.errs_[target_i] /
                             self.utils_[source_i]).numpy()
                        )
                        if self.verbose >= 2:
                            print(
                                "--> actual crit factor for this source/target pair: {:5.5f}".format(factors[-1]))

                        # determine number of "close" neighbors of source unit
                        n_neighbors = np.sum(
                            c_distances[source_i] < self.source_freeze_radius_)-1
                        # freeze those neighbors!!! We won't move any of them now
                        for n_n in sort_c_dist_ind[source_i][1:n_neighbors+1]:
                            frozen[n_n] = True
                        if 0:
                            # determine number of "close" neighbors of target unit
                            n_neighbors = np.sum(
                                c_distances[target_i] < self.target_freeze_radius_)-1
                            # freeze those neighbors!!! We won't move any further one there
                            for n_n in sort_c_dist_ind[target_i][1:n_neighbors+1]:
                                # freeze target neighbor
                                frozen[n_n] = True

                        # consider next target unit
                        target_cur += 1
                        if len(moves) >= self.max_tunnel_moves_per_iter:
                            if self.verbose >= 2:
                                print("not more than {:d} moves per tunnel iter!!!!!!".format(
                                    self.max_tunnel_moves_per_iter))
                            break
        return moves, factors, frozen, to_move

    # NOSONAR
    def _get_sse_and_nearest_centroids(self, X, centroids, direct_call_from_fit=False):
        """compute sse and - for each data point - the nearest centroid

        returns:

        * sse
        * nearest_centroids: list containing nearest centroid for each data point"""
        # shape is int vector in tf 2.0 but Dimension in tf1.4
        n = KMeansTF._shape(X)[0]
        d = KMeansTF._shape(X)[1]
        k = KMeansTF._shape(centroids)[0]

        matrix_size = n*d*k*self._numsize()  # float is e.g. 4 byte
        ready = False
        while not ready:
            if matrix_size <= self.max_mem:
                #
                # calculate with full matrices (for smaller data sets)
                #
                self.fract_ = False  # no splitting of the data set
                expanded_vectors = tf.cast(tf.expand_dims(X, 0),self.__class__._num_type_tf)  # [1,n,d]
                expanded_centroids = tf.cast(tf.expand_dims(centroids, 1),self.__class__._num_type_tf)  # [k,1,d]

                try:
                    # this internally creates matrix [k,n,d] of self._get_num_type()
                    # e.g. k=100 n=1000000 d=10 ==> 10**9*4 bytes = 4GB
                    #
                    # distances is kxn matrix containing the distance of
                    # each (1 of n) data point to each (1 of k) cluster center
                    #
                    distances = tf.reduce_sum(tf.square(
                        tf.subtract(expanded_vectors, expanded_centroids)), 2)  # [k,n]
                    sse = tf.reduce_sum(tf.reduce_min(distances, 0)).numpy()
                    # nearest centroid for all data points
                    nearest_centroids = tf.argmin(distances, 0)  # [n]
                    # nearest samples for all centroids (needed in case of dead units)
                    if direct_call_from_fit:
                        # we need this to handle possible dead units, which do no not occur later (TODO ?)
                        self.nearest_samples = tf.argmin(distances, 1)  # [n]
                    distances = None
                    ready = True
                except Exception as e:
                    #
                    # matrix too large to fit into memory
                    #
                    # sys.exc_info(): the values returned are (type, value, traceback)
                    exc_type, _, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    if self.verbose >= 1:
                        print("Exception ...:", exc_type, fname, exc_tb.tb_lineno)
                    # allocation of matrix_size bytes failed
                    print(f"matrix too large for GPU: bytes {matrix_size:,d}, k={k} n={n:,} d={d} numsize={self._numsize()}, max_mem={self.max_mem:,}")
                    print("automated slicing started")
                    if self.verbose < 1:
                        print("set 'verbose' to 1 to see more details")
                    if self.verbose >= 1:
                        print(e)
                    self.max_mem = matrix_size-1
                    if self.verbose >= 1:
                        print("max_mem is now: {:,d}".format(self.max_mem),
                        " (use the largest working value for the KMeansTF constructor)")
                    continue  # with new self.max_mem value
            else:
                #
                # partition the data set into multiple subsets of equal size
                # (last one is possibly smaller)
                # this is for data sets so large that the required matrices do not
                # fit in GPU memory (adapted to GTX 1060 6MB only, sorry :-))
                # for other cards it my be useful to set self.max_mem differently
                #
                self.fract_ = True
                # number of fractions needed such that each is not larger than self.max_mem
                n_frac = math.ceil(1.0*matrix_size/self.max_mem)
                sse = 0
                nearest_centroids = None
                # loop of partitions of data set
                for f in range(n_frac):
                    # lower index of fraction
                    b1 = f*n//n_frac
                    # higher indes of fraction
                    if f == n_frac-1:
                        # last partition, take the remaining data
                        b2 = n
                    else:
                        # any partition but the last one
                        b2 = (f+1)*n//n_frac

                    expanded_vectors = tf.expand_dims(
                        X[b1:b2], 0)  # [1,n//n_frac,d]
                    expanded_centroids = tf.expand_dims(
                        centroids, 1)  # [k,1,d]
                    frac_size = (b2-b1)*d*k*4
                    if self.verbose > 0 and f == 0:
                        print("frac_size={:,d}".format(frac_size))
                    # this internally creates matrix [k,n//n_frac,d] of self._get_num_type()
                    try:
                        self.verbose > 0 and print(
                            "get_sse_and_nearest_centroids(): handling fraction", f, "of", n_frac)
                        distances_f = tf.reduce_sum(tf.square(
                            tf.subtract(expanded_vectors, expanded_centroids)), 2)  # [k,n//n_frac]
                    except Exception as e:
                        # allocation of frac_size bytes failed
                        print(f"fraction matrix too large for GPU: bytes {frac_size:,d}, k={k} n={b2-b1:,} d={d} numsize={self._numsize()}, max_mem={self.max_mem:,}")
                        self.max_mem = int(frac_size*0.9)
                        if self.verbose >= 1:
                            print(e)
                            print("max_mem is now: {:,d}".format(
                                self.max_mem), " (use the largest working value for the KMeansTF constructor)")
                        break
                    # compute partial SSE
                    sse_f = tf.reduce_sum(
                        tf.reduce_min(distances_f, 0)).numpy()
                    sse += sse_f
                    # compute partial list of nearest_centroids
                    nearest_centroids_f = tf.argmin(distances_f, axis=0)  # [n]
                    # nearest samples for each centroid, however only for the  current fraction
                    # but no problem, we accept that :-)
                    if f == 0 and direct_call_from_fit:
                        self.nearest_samples = tf.argmin(distances_f, 1)  # [k]

                    distances_f = None  # free?
                    if nearest_centroids is None:
                        nearest_centroids = nearest_centroids_f
                    else:
                        nearest_centroids = tf.concat(
                            [nearest_centroids, nearest_centroids_f], axis=0)
                else:
                    # loop over fractions did end normally
                    ready = True
        self.labels_ = nearest_centroids
        return sse, nearest_centroids

    def _get_sse(self, X, centroids):
        """get summed squared error for given samples and centroids"""
        sse, _ = self._get_sse_and_nearest_centroids(X, centroids)
        return sse

    def _get_nearest_centroids(self, X, centroids):
        """get nearest centroid for each data point"""
        _, nearest_centroids = self._get_sse_and_nearest_centroids(X, centroids)
        return nearest_centroids

    def _update_centroids(self, X, nearest_indices, n_clusters):
        """compute new centroids as the mean of all samples associated with a centroid.
        nearest_indices: indicates for each data set the number of the closest centroid"""

        # tf.dynamic_partition requires tf.int32 for second argument!
        nearest_indices = tf.cast(nearest_indices, tf.int32)
        # determine for each centroid the set of signals for which it is nearest
        partitions = tf.dynamic_partition(X, nearest_indices, n_clusters)
        #
        # no partition should be empty!
        #
        # move each centroid to center of gravity of associated set
        new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(
            partition, 0), 0) for partition in partitions], 0)
        return new_centroids

    @staticmethod
    def set_random_seed(seed):        
        """setting random seed for tensorflow, python and numpy
        
        :param seed (int): random seed""" 
        TF2 = tf.__version__[0] == "2"
        if TF2:
            tf.random.set_seed(seed)
        else:
            tf.compat.v1.set_random_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _assert_gpu_memory():
        s= KMeansTF.get_system_status()
        limit_perc = 15
        limit_bytes = 800_000_000
        if (s["gpu_is_used"]==True) \
            and (s["gpu_mem_free_perc"] < limit_perc) \
            and (s["gpu_mem_free_perc"] < limit_bytes):
            raise InsufficientGpuMemoryError(f"not enough free GPU memory: {s['gpu_mem_free_perc']}% ({s['gpu_mem_free']:,} bytes).\n"
        f"required: {limit_perc}% and {limit_bytes:,} bytes.\n"
        "Either free sufficient GPU memory and restart or run kmeanstf without GPU by setting\n"
        '     os.environ["CUDA_VISIBLE_DEVICES"]=""') 

    @staticmethod
    def get_system_status(do_print=False):
        """
        print tensorflow version and availability of GPUs. 

        Args:

            do_print (bool): also print the result

        Example output (if do_print==True)::

            TENSORFLOW: 2.0.0
            Physical GPUs: 1   Logical GPUs: 1
        
        Returns:

            dict with tensorflow version, no of physical GPUs, number of logical GPUs
        """
        #tf_version
        #tf_logi_cpus
        #tf_phys_cpus
        #gpu_mem_total
        #gpu_mem_free
        #gpu_mem_used
        #nvidia_gpu
        #gpu_is_used
        #
        #
        # pylint: disable=maybe-no-member
        if do_print:
            print("TENSORFLOW:", tf.__version__)
        ret = {
            "tf_version":None,
            "tf_logi_gpus":None,
            "tf_phys_gpus":None,
            "gpu_mem_total":None,
            "gpu_mem_free":None,
            "gpu_mem_used":None,
            "gpu_mem_free_perc":None,
            "nvidia_gpu":None,
            "gpu_is_used":None,
        }
        ret["tf_version"]=tf.__version__
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                # for gpu in gpus:
                #    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                ret["tf_phys_gpus"] = len(gpus)
                ret["tf_logi_gpus"] = len(logical_gpus)
                if do_print:
                    print("Physical GPUs:", ret["tf_phys_gpus"],
                        "  Logical GPUs:", ret["tf_logi_gpus"])
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        else:
            if do_print:
                print("no Tensorflow-visible GPUs available")
            ret["tf_phys_gpus"] = 0
            ret["tf_logi_gpus"] = 0

        try:
            nvidia_smi.nvmlInit()
        except:
            ret["nvidia_gpu"] = False
            ret["gpu_mem_total"] = 0
            ret["gpu_mem_free"] = 0
            ret["gpu_mem_used"] = 0
        else:
            ret["nvidia_gpu"] = True

            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

            if do_print:
                print(f"Total GPU Memory: {info.total:,d}")
                print(f"Free GPU Memory: {info.free:,d}")
                print(f"Used GPU Memory: {info.used:,d}")
            ret["gpu_mem_total"] = info.total
            ret["gpu_mem_free"] = info.free
            ret["gpu_mem_used"] = info.used
            ret["gpu_mem_free_perc"] = round(ret["gpu_mem_free"]/ret["gpu_mem_total"]*100,2)

            nvidia_smi.nvmlShutdown()   
        ret["gpu_is_used"] = ret["tf_logi_gpus"]!=0     
        return ret
    
    @staticmethod
    def _shape(t):
        # pylint: disable=maybe-no-member
        TF2 = tf.__version__[0] == "2"
        if TF2:
            # shape is int vector in tf 2.0
            return(t.shape)
        else:
            # shape is TensorShape object in tf 1.x
            # with Dimension as fields
            return t.shape.as_list()

    @staticmethod
    def _ugauss(n, d, sigma):
        """generate (n,d)-tensor of N(0,sigma)-distributed values"""
        return tf.random.normal(
            (n, d),
            mean=0.0,
            stddev=sigma,
            dtype=TunnelKMeansTF._get_num_type()
        )

    @staticmethod
    def get_gaussian_mixture(n=1000, d=2, g=50, sigma=0.0005):
        """generate test data from Gaussian mixture distribution
        
        Returns (n,d)-Tensor from mixture of g Gaussians with standard deviation *sigma*."""
        centers = tf.random.uniform((g, d),dtype=TunnelKMeansTF._get_num_type())
        return tf.concat([KMeansTF._ugauss(int(np.ceil(n/g)), d, sigma)+centers[i] for i in range(g)], axis=0)[0:n]

    # general plotting function for one kmeansX result    
    @staticmethod
    def _km_plot(ax,  # NOSONAR
            X=None, # data set
            C=None, # codebook
            cap=None, # caption
            ticks=False, # have ticks?
            fontsize=16, # fontsize for caption
            dotsize=2, # size of data points
            centersize=3, # size of centroid
            ypos=1.02, # position for caption
            voro=False, # voronoi diagram
            **kwargs): # catchall

        """
        plots:
        * data set
        * caption
        * centers (rims around centers with a suitable color)
        * possibly select 2 of d dimensions
        * Voronoi diagrams
        """

        cocy = ['green', 'red', 'yellow']

        # Voronoi diagram wrt centers
        if voro and C is not None and C.shape[0]>2:
            vor = Voronoi(C)
            voronoi_plot_2d(vor,ax=ax, show_vertices=False, show_points=True,line_colors="blue")
            
        # data
        if X is not None:
            ax.set_prop_cycle(color=cocy)
            ax.plot(X[:,0],X[:,1],".", ms=dotsize,zorder=-10)

        # codebook
        if C is not None:
            ax.plot(C[:,0],C[:,1],"o", ms=centersize,color="red",zorder=9)
    
        # ticks
        if not ticks:
            ax.tick_params(length=0)
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])

        # caption
        if cap is not None:
            plt.text(0.5, ypos, cap,
            horizontalalignment='center',
            fontsize=fontsize,
            transform = ax.transAxes)
        
        ax.set_aspect(1.0)


    @staticmethod
    def _plot(ax,X,C=None,cap=None, over=None, voro=1):
            KMeansTF._km_plot(ax,X,C=C,cap=cap,voro=voro)
            #return
            ax.set_aspect(1)
            ax.set_xlim((-0.1,1.1))
            ax.set_ylim((-0.1,1.1))
            if not over is None:
                ax.text(0.5, 0.98, over, horizontalalignment='center', verticalalignment='top', 
                transform=ax.transAxes, fontsize=25, fontweight="bold", alpha=0.4)



    @staticmethod
    def self_test(X=None,  n_clusters=100, n_init=10, n=10000, 
    d=2, g=50, sigma:float=None, verbose=0, stats_only=0, 
    init="k-means++",plot=True, voro=True):
        """self-testing routine
        
        runs both k-means++ and tunnel k-means and prints the SSE improvement 
        of tunnel k-means over k-means++ (in the scikit-learn implementation). 
        Uses Gaussian mixture distribution (default) or provided data set *X*. Typical output::

            Data is mixture of 50 Gaussians in unit square with sigma=0.00711
            algorithm      | data.shape  |   k  | init      | n_init  |     SSE   | Runtime  | Improvement
            ---------------|-------------|------|-----------|---------|-----------|----------|------------
            k-means++      | (10000, 2)  |  100 | k-means++ |      10 |   0.66179 |    2.09s | 0.00%
            tunnel k-means | (10000, 2)  |  100 | random    |       1 |   0.63933 |    3.37s | 3.39%

    Args:
        X: data set to use (as tensorflow or numpy array). If None, use mixture of Gaussians according to the other parameters
        n_clusters (int): the *k* in *k*-means
        n_init (int): number of runs with different initializations
        n (int): number of data points to generate
        d (int): number of features (dimensionality) of generated data points
        g (int): number of Gaussians
        sigma (float): standard deviation of Gaussians, if 'None' a value is chosen based on number of Gaussians
        init ('k-means++' or 'random'): initialization method for *k*-means (tunnel *k*-means is initialized as random)
        plot (bool): plot the result?
        voro (bool): show Voronoi regions in plot?
        
        
        """
        print("self test ...")
        if sigma is None:
            sigma=1.0/(7+math.sqrt(g))/10*2
        # create data set
        if X is None:
            X = KMeansTF.get_gaussian_mixture(n=n, d=d,g=g, sigma=sigma)
            if d == 1:
                box="segment"
            if d == 2:
                box="square"
            if d == 3:
                box="cube"
            if d > 3:
                box=f"{d}D-hypercube"
            print(f"Data is mixture of {g} Gaussians in unit {box} with sigma={sigma:.5f}")
        else:
            print(f"Data is provided by caller. Statistial properties unknown")
        print(f"algorithm      | data.shape  |   k  | init      | n_init  |     SSE   | Runtime  | Improvement")
        print(f"---------------|-------------|------|-----------|---------|-----------|----------|------------")


        kms = KMeans(n_clusters=n_clusters, n_init=n_init,
                      verbose=verbose, init=init, algorithm="full")
        #
        # sklearn.KMeans has ‘elkan’ as default which as of Jan 2020 interprets ‘tol’-Parameter differently 
        # than ‘full’, see https://github.com/scikit-learn/scikit-learn/issues/15831
        # which leads to different numbers of Lloyd iterations
        # to make results comparable, we use "full" here
        start_kms = time()
        kms.fit(X)
        t_kms = time()-start_kms
        CKMS=kms.cluster_centers_

        imp = 0
        print(f"k-means++      | {str(X.shape):11s} | {n_clusters:4d} | {init:9s} | {n_init:7d} | {kms.inertia_:9.5f} | {t_kms:7.2f}s | {imp:.2%}")
        if 0:
            km = KMeansTF(n_clusters=n_clusters, n_init=n_init,
                        verbose=verbose, init=init)
            start_kmpp = time()
            km.fit(X)
            t_kmpp = time()-start_kmpp
            imp = (kms.inertia_-km.inertia_)/kms.inertia_
            print(f"k-means++  | {str(X.shape):11s} | {n_clusters:4d} | {init:9s} | {n_init:7d} | {km.inertia_:9.5f} | {t_kmpp:7.2f}s | {imp:.2%}")


        init="random"
        tkm1 = TunnelKMeansTF(n_clusters=n_clusters,init=init, verbose=verbose)
        start_tkm1 = time()
        tkm1.fit(X)
        t_tkm1 = time()-start_tkm1
        imp1 = (kms.inertia_-tkm1.inertia_)/kms.inertia_
        n_init=1
        print(f"tunnel k-means | {str(X.shape):11s} | {n_clusters:4d} | {init:9s} | {n_init:7d} | {tkm1.inertia_:9.5f} | {t_tkm1:7.2f}s | {imp1:.2%}")
        tunnel_iters = tkm1.tunnel_history
        CTKM=tkm1.cluster_centers_

        if 0:
            init="k-means++"
            tkm2 = TunnelKMeansTF(n_clusters=n_clusters,init="k-means++", verbose=verbose)
            start_tkm2 = time()
            tkm2.fit(X)
            t_tkm2 = time()-start_tkm2
            imp2 = (kms.inertia_-tkm2.inertia_)/kms.inertia_
            n_init=1
            print(f"tunnel k-means | {str(X.shape):11s} | {n_clusters:4d} | {init:9s} | {n_init:7d} | {tkm2.inertia_:9.5f} | {t_tkm2:7.2f}s | {imp2:.2%}")
            tunnel_iters = tkm2.tunnel_history

        if plot:
            import matplotlib.pyplot as plt
            fig,axs=plt.subplots(1,2,figsize=(14,7))
            KMeansTF._plot(axs[0],X,C=CKMS, 
            cap=f"k-means++, SSE: {kms.inertia_:.5f}",voro=voro)
            if kms.inertia_ >= tkm1.inertia_:
                evalu = "lower"
            else:
                evalu = "higher (oops!)"
            KMeansTF._plot(axs[1],X,C=CTKM, 
            cap=f"tunnel k-means, SSE: {tkm1.inertia_:.5f}", over=f"SSE {imp1:.2%} {evalu}",voro=voro)
            #fig.tight_layout()
            plt.show()

        flatten = itertools.chain.from_iterable
        tunnel_moves = list(flatten(tunnel_iters))
        if 0:
            return {
                "n": KMeansTF._shape(X)[0],
                "d": KMeansTF._shape(X)[1],
                "k": n_clusters,
                "sse_kmpp": km.inertia_,
                "sse_tkm": tkm.inertia_,
                "improvement_percent": imp*100,
                "t_kmpp": t_kmpp,
                "t_tkm": t_tkm,
                "km.lloyd": km.no_of_lloyd,
                "tkm.lloyd": tkm.no_of_lloyd,
                "tunnel_iters": len(tunnel_iters),
                "tunnel_moves": len(tunnel_moves),
                "km": tkm,
                "data": X
            }

class KMeansTF(BaseKMeansTF):
    """ implements k-means/k-means++

For full desription of methods see base class :class:`BaseKMeansTF`

Args:
    n_clusters (int): The number of clusters to form as well as the number of centroids to generate.
    init ('k-means++','random' or array): method of initialization

    n_init (int):     number of runs with different initializations (default 10)
    max_iter (int):   Maximum number of Lloyd iterations for a single run of the k-means algorithm.
    tol (float):        Relative tolerance with regards to inertia to declare convergence.
    verbose (int):    Verbosity mode.
    random_state (int): None, or integer to seed the random number generators of python, numpy and tensorflow

Attributes:
    cluster_centers_ (array, [n_clusters, n_features]): Coordinates of cluster centers. If the algorithm stops before fully 
        converging (see tol and max_iter), these will not be consistent with labels\\_.
    labels_ (array, shape(n_samples)): Labels of each point, i.e. index of closest centroid
    inertia_ (float): Sum of squared distances of samples to their closest cluster center.
    n_iter_ (int): Number of iterations run.
    """

    def __init__(self, n_clusters=8, init='k-means++', n_init:int=10, max_iter=300, tol=1e-4, verbose=0, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state)


class TunnelKMeansTF(BaseKMeansTF):
    """ implements tunnel k-means

For full desription of methods see base class :class:`BaseKMeansTF`

Args:
    n_clusters (int): The number of clusters to form as well as the number of centroids to generate.
    init ('random', 'k-means++' or array): method of initialization
    n_init (int): number of runs of the *initial* k-means phase with different initializations (default 1). 
                    Only **one** tunnel phase is performed even if n_init is larger than 1.
    max_iter (int):   Maximum number of Lloyd iterations for a single run of the k-means algorithm.
    tol (float): Relative tolerance with regards to inertia to declare convergence.
    verbose (int):    Verbosity mode.
    random_state (int): None, or integer to seed the random number generators of python, numpy and tensorflow
    max_tunnel_iter (int): how many tunnel iterations to perform maximally
    max_tunnel_moves_per_iter (int): how many centroids to move maximally in one tunnel iteration
    criterion (float): inital required ratio error/utility (is increased adaptively)
    local_trials (int): how many time should each tunnel move be repeated with different random offset vector (1 or larger)
    collect_history (bool): collect historic information on inertia, criterion, tunnel moves, codebooks

Attributes:
    cluster_centers_ (array, [n_clusters, n_features]): Coordinates of cluster centers. If the algorithm stops before fully 
        converging (see tol and max_iter), these will not be consistent with labels\\_.
    labels_ (array, shape(n_samples)): Labels of each point, i.e. index of closest centroid
    inertia_ (float): Sum of squared distances of samples to their closest cluster center.
    n_iter_ (int): Number of iterations run.

    """

    def __init__(self,  # NOSONAR
                 n_clusters=8,
                 init='random',
                 n_init=1,
                 max_iter=300,
                 tol=1e-4,
                 verbose=0,
                 random_state=None,
                 max_tunnel_iter=300,
                 max_tunnel_moves_per_iter=100,
                 criterion=1.0,
                 local_trials=1,
                 collect_history=False):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            tunnel=True,
            max_tunnel_iter=max_tunnel_iter,
            max_tunnel_moves_per_iter=max_tunnel_moves_per_iter,
            criterion=criterion,
            local_trials=local_trials,
            collect_history=collect_history)
