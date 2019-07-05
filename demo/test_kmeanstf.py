import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from time import time

from sklearn.cluster import KMeans
from kmeanstf import KMeansTF
n = 1
n_clusters = 100
n_init = 1
tol=1e-4
tol=0
max_iter=30
init = 'k-means++'

# create data set
#X = tf.random.normal([1000000,2])
#X = tf.random.normal([3000000,2])
X = tf.random.normal([10000,2])

print("**** Tensorflow-based k-means++ ****")
s = KMeansTF(n_clusters=n_clusters, n_init=n_init, init=init, tol=tol, max_iter=max_iter,max_mem=1300000000)
start = time()
pred = s.fit_predict(X)
print(pred[:10])
print("1st vector", X[0])
print("predicted centroid:",s.cluster_centers_[pred[0]])
d1=time()-start
print("duration: {:5.2f}s n_iter: {:} ".format(d1, s.n_iter_))
sse1 = s.inertia_

comp = True # compare to sklearn
#comp = False
if comp:
    print("**** k-means++ from scikit-learn (please be patient) ****")
    km = KMeans(n_clusters=n_clusters, n_init=n_init, init=init, tol=tol, max_iter=max_iter, algorithm="elkan")
    start = time()
    pred = km.fit_predict(X)
    print(pred[:10000])
    print("1st vector", X[0])
    print("predicted centroid:",km.cluster_centers_[pred[0]])
    end = time()
    d2=time()-start
    print("duration: {:5.2f}s n_iter: {:} ".format(d2, km.n_iter_))
    sse2 = km.inertia_

# figure
if comp:
    fig,(ax,ax2) = plt.subplots(1,2, figsize=(16,8))
else:
    fig,ax = plt.subplots(1,1,figsize=(8,8))

ax.set_aspect('equal')
c = s.cluster_centers_
m = 10000 # maxdraw
ax.scatter(X[:m,0],X[:m,1],s=1)
ax.scatter(c[:,0],c[:,1],s=5**2,c="w")
ax.scatter(c[:,0],c[:,1],s=4**2,c="r")
title = "tf-gpu:  iter={:2d}  t={:5.2f}s  SSE={:6.1f}".format( s.n_iter_, d1,sse1)
if comp:
    title = title+" speed-up: {:5.2f}".format(d2/d1)
ax.set_title(title)
if comp:
    ax=ax2
    ax.set_aspect('equal')
    c = km.cluster_centers_
    ax.scatter(X[:m,0],X[:m,1],s=1)
    ax.scatter(c[:,0],c[:,1],s=5**2,c="w")    
    ax.scatter(c[:,0],c[:,1],s=4**2,c="r")
    ax.set_title("scikit-learn:  iter={:2d}  t={:5.2f}s  SSE={:6.1f}".format( km.n_iter_, d2,sse2))

#print("tippi:",type(X.shape[0]),"<===>",type(c.shape[0]))
if isinstance(X.shape[0], int):
    npoints = X.shape[0]
else:
    npoints = X.shape[0].value
fig.suptitle("k-means++:  points: {:,} (only {:,d} shown)  centers: {:3}".
format( npoints, m, n_clusters, fontsize=16))
plt.show()