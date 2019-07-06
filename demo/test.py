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
