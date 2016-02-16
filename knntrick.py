import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import numpy as np

def evaluate(nbrs, X, m):
    nbrs.fit(X)
    s = time.time()
    indices = []
    for i in range(0,m):
	indices.extend(nbrs.kneighbors([X[i]], return_distance=False)[0])
    e = time.time()
    return np.array(indices), e-s


def evaluateAll(nbrs, X, m):
    nbrs.fit(X)
    s = time.time()
    _,indices = nbrs.kneighbors(X[:m], return_distance=True)
    e = time.time()
    return np.array(indices), e-s

m=20


fast = []
fast2 = []
slow = []
ns = xrange(5000, 20000, 1000)
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

    nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean', algorithm='ball_tree')
    indices2, t = evaluate(nbrs,X,m)
    assert (indices1==indices2).all(), "ooops, solutions don't match, they should."
    fast.append(t)

    nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean', algorithm='kd_tree')
    indices3, t = evaluate(nbrs,X,m)
    assert (indices1==indices3).all(), "ooops, solutions don't match, they should."
    fast2.append(t)



    print n

plt.scatter(ns, slow, color='blue', label='cosine brute force')
plt.scatter(ns, fast, color='red', label ='norm euclidean ball tree')
plt.scatter(ns, fast2, color='green', label='norm euclidean kd tree')
plt.legend()
plt.show()

