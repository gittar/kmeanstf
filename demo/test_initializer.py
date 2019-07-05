import numpy as np
import tensorflow as tf
import sys
import time
import matplotlib.pyplot as plt
from kmeanstf import Initializer
def main1():
    """ minimal test: create data, initialize codebook, plot
    """
    print("main1 example")
    n = 2000000
    n = 2000000
    d = 2
    k = 100
    draw_size=4000

    methods = "random k-means++".split(" ")
    data = tf.random.normal((n,d))

    codebooks = [None]*len(methods)
    times = [None]*len(methods)
    sse = [None]*len(methods)

    for i,method in enumerate(methods):
        start = time.time()
        codebooks[i],sse[i]=Initializer.init(data,k,method)
        times[i]=time.time()-start
        print("jiji",sse[i])

    fig,axes = plt.subplots(nrows=1, ncols=len(methods), sharex=True,sharey=True, figsize=( 12,5))
    if len(methods) == 1:
        axes = [axes]
    for ax,cb,method, t in zip(axes,codebooks,methods, times):
        ax.set_aspect('equal')
        ax.scatter(data[:np.minimum(n, draw_size),0], data[:np.minimum(n, draw_size),1], s=0.5,c="k")
        ax.scatter(cb[:,0], cb[:,1], s=5,c="r")
        ax.set_title(method+ " {:5.3f}s".format(t))

    plt.show()


if len(sys.argv)>1:
    print (sys.argv[1])
print (sys.argv)
main1()