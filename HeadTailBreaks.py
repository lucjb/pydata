__author__ = 'lbernardi'
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

def htb(data, tails=[]):
    if len(data)==0:
        return tails
    mean = sum([size for _, size in data])/len(data)
    head = []
    tail = []
    for x, size in data:
        if size > mean:
            head.append((x, size))
        else:
            tail.append((x, size))

    if len(head)>=len(data)*.5:
        return tails

    tails.append(tail)
    return htb(head, tails)

def head_tail_breaks(data):
    tails =  htb(data)
    breaks = []
    for tail in tails:
        x, size = zip(*tail)
        breaks.append(np.min(size))

    return sorted(breaks)

def head_tail_breaks_encode(X, col):
    values = sorted(X[:,col], reverse=True)
    rank_size = zip(range(0,len(values)), values)
    breaks = head_tail_breaks(rank_size)
    digitized_values = np.digitize(values, bins=breaks)
    digitized_values = digitized_values.reshape((-1,1))
    enc = OneHotEncoder(sparse=False)
    ohe_values =  enc.fit_transform(digitized_values)
    X_new = np.append(X, ohe_values, axis=1)
    return X_new

boston = load_boston()

X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
print lr.score(X_test, y_test)
plain_rmse = mean_squared_error(y_test, lr.predict(X_test))
print plain_rmse


boston = load_boston()
X, y = boston.data, boston.target

selected = []
baseline_rmse = plain_rmse

for col, fn in enumerate(boston.feature_names):
    X_new = head_tail_breaks_encode(X, col)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.3, random_state=0)
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)   
    rmse = mean_squared_error(y_test, lr.predict(X_test))
    r = lr.score(X_test, y_test)
    if rmse < baseline_rmse:
	print fn, rmse, r
	baseline_rmse = rmse
    	values = sorted(X[:,col], reverse=True)
	plt.plot(range(0,len(values)), values)
	plt.show()
	selected.append(col)
    

print '='*70
boston = load_boston()
X, y = boston.data, boston.target

for col in selected:
    X = head_tail_breaks_encode(X, col)    


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)   
rmse = mean_squared_error(y_test, lr.predict(X_test))
r = lr.score(X_test, y_test)
print rmse, r
print 'Improvement: %f%%' % ((1-rmse/plain_rmse)*100)

