import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.utils import shuffle

def evaluate(nbrs, X, m):
    sf = time.time()
    nbrs.fit(X)
    ef = time.time()
    s = time.time()
    indices = []
    for i in range(0,m):
       knn = nbrs.kneighbors([X[i]], return_distance=False)[0]
       indices.extend(knn)
    e = time.time()
    return np.array(indices), e-s, ef-sf



def evaluateAll(nbrs, X, m):
    nbrs.fit(X)
    s = time.time()
    _,indices = nbrs.kneighbors(X[:m], return_distance=True)
    e = time.time()

    return np.array(indices), e-s

def plot_knn(X):
    f, axarr = plt.subplots(6, 10)	
    for i in range(0,6):
       knn = nbrs.kneighbors([X[i]], return_distance=False)[0]
       r=0
       for p in knn:
           axarr[i,r].matshow(X[p].reshape((28,28)))
           axarr[i,r].set_xticks([])		
           axarr[i,r].set_yticks([])		
           r+=1
    plt.tight_layout()
    plt.show()


k = 10
mnist = fetch_mldata("MNIST original")
mnist.data, mnist.target = shuffle(mnist.data, mnist.target)
plt.matshow(mnist.data[0].reshape((28,28)))
plt.show()

nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
nbrs.fit(mnist.data)
plot_knn(mnist.data)

X=[]
for x in mnist.data:
	x = x.astype(float)
	X.append(x)

new_shape=X[1].shape[0]

for i,x in enumerate(X):
	X[i] = normalize(x).reshape((new_shape))

nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean', algorithm='ball_tree')
nbrs.fit(X)
plot_knn(X)

m = 20

fast = []
fast2 = []
slow = []

fast_fit = []
fast2_fit = []
slow_fit = []

s, e, delta = 3900, 4000, 100
ns = xrange(s, e, delta)

for n in ns:
    
    mnist.data, mnist.target = shuffle(mnist.data, mnist.target)

    X=[]
    for x in mnist.data[:n]:
        x = x.astype(float)
        X.append(x)

    new_shape=X[1].shape[0]

    for i,x in enumerate(X):
        X[i] = normalize(x).reshape((new_shape))

    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    indices1, t, t_f = evaluate(nbrs,X,m)
    slow.append(t)
    slow_fit.append(t_f)

    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='ball_tree')
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
plt.legend(loc='upper left')
plt.title('kNN look up time vs data set size')
plt.xlabel('n')
plt.ylabel('time')
plt.tight_layout()
plt.show()

plt.scatter(ns, slow_fit, color='blue', label='cosine brute force')
plt.scatter(ns, fast_fit, color='red', label ='norm euclidean ball tree')
plt.scatter(ns, fast2_fit, color='green', label='norm euclidean kd tree')
plt.legend(loc='upper left')
plt.title('kNN fit time vs data set size')
plt.xlabel('n')
plt.ylabel('time')
plt.tight_layout()
plt.show()
