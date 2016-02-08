import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import numpy as np

def evaluate(nbrs, X, m):
    nbrs.fit(X)
    s = time.time()
    indices = None
    for i in range(0,m):
        indices = nbrs.kneighbors([X[i]], return_distance=False)
    e = time.time()
    return indices, e-s

m=10


fast = []
slow = []
ns = xrange(1000, 10000, 1000)
for n in ns:
    mnist = fetch_mldata("MNIST original")
    y = mnist.target[:n]

    X=[]
    for x in mnist.data[:n]:
        x = x.astype(float)
        X.append(x)


    nbrs = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    indices1, t = evaluate(nbrs,X,m)
    slow.append(t)


    for i,x in enumerate(X):
        X[i] = normalize(x).reshape((784))
    nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean', algorithm='kd_tree')

    indices2, t = evaluate(nbrs,X,m)
    assert (indices1==indices2).all(), "ooops, solutions don't match, they should."
    fast.append(t)
    print n

plt.scatter(ns, slow)
plt.scatter(ns, fast, color='red')
plt.show()

