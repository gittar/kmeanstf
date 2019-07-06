import tensorflow as tf

from kmeanstf import KMeansTF
n = 1
n_clusters = 100
n_init = 1
tol=1e-4
tol=0
max_iter=30
init = 'k-means++'

if tf.test.is_gpu_available():
    print("GPU is present")
else:
    print("no GPU is present!")

if tf.test.is_built_with_cuda():
    print("tf is built with cuda")
else:
    print("tf is not built with cuda")

# create data set

X = tf.random.normal([1000,2])

print("**** testing ****")
s = KMeansTF(n_clusters=n_clusters, n_init=n_init, init=init, tol=tol, max_iter=max_iter,max_mem=1300000000)
s.fit(X)
print("fit", s.inertia_)
labels = s.fit_predict(X)
print("fit_predict", s.inertia_, labels)
labels =s.predict(s.cluster_centers_)
print("predict", s.inertia_, labels)

