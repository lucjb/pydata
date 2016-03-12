__author__ = 'lbernardi'
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

def plot_htb(values, breaks):
    breaks.append(min(values))
    breaks.append(max(values))
    breaks.sort()
    plt.title('CRIM: per capita crime rate by town')
    plt.xlabel('House Ranking Position')

    plt.plot(values)
    plt.show()
    f, _ = np.histogram(values, bins=breaks)
    plt.hist(values, bins=breaks, orientation='horizontal')

    plt.plot(values, color='red')
    plt.title('CRIM: per capita crime rate by town')
    plt.xlabel('House Ranking Position')
    plt.xticks(f)
    plt.yticks(breaks)
    plt.grid()
    plt.show()


def htb(data, tails=[], breaks=[]):
    if len(data)==0:
        return tails, breaks
    mean = sum(zip(*data)[1])/len(data)
    head = []
    tail = []
    for x, size in data:
        if size > mean:
            head.append((x, size))
        else:
            tail.append((x, size))

    if len(head)>=len(data):
	return tails, breaks

    tails.append(tail)
    breaks.append(mean)
    return htb(head, tails, breaks)

def head_tail_breaks(data):
    tails, breaks =  htb(data, [], [])
    return breaks

def head_tail_breaks_encode(X, col):
    values = sorted(X[:,col], reverse=True)
    rank_size = zip(range(1,len(values)+1), values)
    breaks = head_tail_breaks(rank_size)
    if len(breaks)<1:
	return X, breaks
    digitized_values = np.digitize(values, bins=breaks)
    digitized_values = digitized_values.reshape((-1,1))
    enc = OneHotEncoder(sparse=False)
    ohe_values =  enc.fit_transform(digitized_values)
    X_new = np.append(X, digitized_values, axis=1)

    return X_new, breaks

#Ladd boston housing data from UCI
boston = load_boston()

X, y = boston.data, boston.target

#Compute baseline model and performance
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
print lr.score(X_test, y_test)
plain_rmse = mean_squared_error(y_test, lr.predict(X_test))
print plain_rmse

# Apply the head tail breaks trick to every column and evaluate the model individually
boston = load_boston()
X, y = boston.data, boston.target
selected = []
baseline_rmse = plain_rmse

for col, fn in enumerate(boston.feature_names):
    X_new, breaks = head_tail_breaks_encode(X, col)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y, test_size=0.3, random_state=0)
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)   
    rmse = mean_squared_error(y_test, lr.predict(X_test))
    r = lr.score(X_test, y_test)
    print fn, rmse, r, breaks
    selected.append(col)
    

print '='*70

# Apply the head tail breaks to every column and evaluate the model using all the features
boston = load_boston()
X, y = boston.data, boston.target

for col, fn in enumerate(boston.feature_names):
    X, _ = head_tail_breaks_encode(X, col)    


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)   
rmse = mean_squared_error(y_test, lr.predict(X_test))
r = lr.score(X_test, y_test)
print rmse, r
print 'Total Improvement: %f%%' % ((1-rmse/plain_rmse)*100)


#Plots thos most interesting feature with breaks and a histogram.
col=0
_, breaks = head_tail_breaks_encode(X, col)
values = sorted(X[:,0],
 reverse=True)
plot_htb(values, breaks)

