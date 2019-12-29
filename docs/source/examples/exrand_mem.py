import numpy as np
import matplotlib.pyplot as plt
from kmeanstf import KMeansTF

km = KMeansTF(n_clusters=100, random_state=1,max_mem = 1000000, verbose=1)
X = np.random.normal(size=(5000,2))
km.fit(X)

plt.scatter(X[:,0],X[:,1],s=1)
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=5,c="r")
plt.title(f"k-means++, SSE = {km.inertia_:.3f}")
plt.show()