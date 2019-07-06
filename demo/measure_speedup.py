"""
find out the actual speed-up of kmeanstf over scikit-learn on your hardware

This program runs both KMeansTF and KMeans from scikit-learn on a sequence of
increasingly larger data sets and increasingly larger values of k

For each problem size the speedup is printed out.

Interrupt if the waiting time becomes too long for the larger problems
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from kmeanstf import KMeansTF
from sklearn.cluster import KMeans
import pandas as pd

if tf.test.is_gpu_available():
    print("GPU is present")
else:
    print("no GPU is present!")

if tf.test.is_built_with_cuda():
    print("tf is built with cuda")
else:
    print("tf is not built with cuda")
    
n = 1
n_clusters = 10
n_init = 1
tol=1e-4
tol=0
max_iter=30
init="k-means++"

mdata =[]
#init = "random"

# d = data dimensionality (no of features)
# n = number of data points
# k = number of centroids

#for n in [10000,100000,300000,1000000,3000000]:
#for n in [1000000, 2000000, 4000000, 8000000]:
#for n in [3000]:
for n in [30000,100000,300000, 1000000,3000000,10000000]:

    #for d in [2,4,6,8,10]:
    for d in [2]:
        # create data set
        X = tf.random.normal([n,d])

        for k in range(10,100,10):
        #for k in range(10,20,10):
            print("d:{:2d} n: {:7d} k:{:4d}".format(d,n,k), end=" ")
            res={}
            for cl in ["KMeansTF", "KMeans"]:
            #for cl in ["KMeansTF"]:
                print(cl, end=" ")
                km = globals()[cl](n_clusters=k, n_init=n_init, init=init, tol=tol, max_iter=max_iter)
                km.fract_ = False
                km.init_duration_ = 0
                start = time()
                km.fit(X)
                end = time()
                res[cl]=time()-start
                print("duration: {:5.2f} iter: {:3d}".format(res[cl], km.n_iter_), end=" ")
                if cl == "KMeansTF":
                    fract = km.fract_
                t = res[cl]
                ti=km.init_duration_
                data=[cl,n,d,k,t,ti,km.n_iter_, km.inertia_, km.fract_]
                mdata.append(data)
            if not "KMeans" in res:
                res["KMeans"]=1
                speedup = 0
            else:
                speedup=res["KMeans"]/res["KMeansTF"]
            
            print("speedup: {:5.2f} fract={:}".format(speedup, fract ))
    print("store for ",n)
    df=pd.DataFrame(mdata,columns="cl n d k t i it mse fr".split(" "))
    #df.to_pickle("./statistics.pkl")




#store data
#
# chart to show speedup for differnt values of k (x-axis) for certain n
# several charts for various n over each other
#

#
# chart to show execution for differnt values of k (x-axis) for certain n
# several charts for various n over each other
#

