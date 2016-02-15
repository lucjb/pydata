import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def evaluate(nbrs, X, m):
    nbrs.fit(X)
    s = time.time()
    indices = None
    for i in range(0,m):
	Xi = X[i].toarray()[0]
        indices = nbrs.kneighbors([Xi], return_distance=False)
    e = time.time()
    return indices, e-s

m=10

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

print("Loading 20 newsgroups dataset for categories:")
print(categories)


fast = []
slow = []
ns = xrange(100, 1000, 100)
for n in ns:

    dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english', use_idf=True)
    X = vectorizer.fit_transform(dataset.data[:n])

    nbrs = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    indices1, t = evaluate(nbrs,X,m)
    slow.append(t)


    for i,x in enumerate(X):
        X[i] = normalize(x)
    nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean', algorithm='kd_tree')

    indices2, t = evaluate(nbrs,X,m)
    assert (indices1==indices2).all(), "ooops, solutions don't match, they should."
    fast.append(t)
    print n

plt.scatter(ns, slow)
plt.scatter(ns, fast, color='red')
plt.show()

