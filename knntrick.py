import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.utils import shuffle#

def evaluate(nbrs, X, m):
    sf = time.time()#
    nbrs.fit(X)
    ef = time.time()#
    s = time.time()
    indices = []
    for i in range(0,m):
       indices.extend(nbrs.kneighbors([X[i]], return_distance=False)[0])#
    e = time.time()
    return np.array(indices), e-s, ef-sf


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

fast_fit = []
fast2_fit = []
slow_fit = []

ns = xrange(5000, 40000, 1000)
mnist = fetch_mldata("MNIST original")
for n in ns:
    
    mnist.data, mnist.target = shuffle(mnist.data, mnist.target)#

    y = mnist.target[:n]

    X=[]
    for x in mnist.data[:n]:
        x = x.astype(float)
        X.append(x)

    new_shape=X[1].shape[0]#

    for i,x in enumerate(X):#
        X[i] = normalize(x).reshape((new_shape))#

    nbrs = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    indices1, t, t_f = evaluate(nbrs,X,m)
    slow.append(t)
    slow_fit.append(t_f)

    nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean', algorithm='ball_tree')
    indices2, t, t_f = evaluate(nbrs,X,m)
    assert (indices1==indices2).all(), "ooops, solutions don't match, they should."
    fast.append(t)
    fast_fit.append(t_f)

    nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean', algorithm='kd_tree')
    indices3, t, t_f = evaluate(nbrs,X,m)
    assert (indices1==indices3).all(), "ooops, solutions don't match, they should."
    fast2.append(t)
    fast2_fit.append(t_f)

    print n

plt.scatter(ns, slow, color='blue', label='cosine brute force')
plt.scatter(ns, fast, color='red', label ='norm euclidean ball tree')
plt.scatter(ns, fast2, color='green', label='norm euclidean kd tree')
plt.legend()
plt.show()

plt.scatter(ns, slow_fit, color='blue', label='cosine brute force')
plt.scatter(ns, fast_fit, color='red', label ='norm euclidean ball tree')
plt.scatter(ns, fast2_fit, color='green', label='norm euclidean kd tree')
plt.legend()
plt.show()
