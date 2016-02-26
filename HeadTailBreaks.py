__author__ = 'lbernardi'
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import numpy as np
from sklearn.gaussian_process import GaussianProcess
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
from sklearn import cross_validation
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

    print len(head), len(data)

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

#2
boston = load_boston()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(boston.data, boston.target, test_size=0.3, random_state=0)
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
print lr.score(X_test, y_test)


for col in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
    print col
    values = sorted(boston.data[:,col], key=lambda x: -x)
    plt.plot(values)
    #plt.show()

    rank_size = zip(range(0,len(values)), values)
    breaks = head_tail_breaks(rank_size)
    print breaks
    digitized_crime_rate = np.digitize(values, bins=breaks)
    digitized_crime_rate = digitized_crime_rate.reshape((-1,1))
    print digitized_crime_rate
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse=False)
    ohe_crime_rate =  enc.fit_transform(digitized_crime_rate)
    plt.hist(values, bins=breaks)
    #plt.show()
    #boston.data = np.delete(boston.data, col, axis=1)
    boston.data = np.append(boston.data, ohe_crime_rate, axis=1)
    print boston.data.shape


X_train, X_test, y_train, y_test = cross_validation.train_test_split(boston.data, boston.target, test_size=0.3, random_state=0)
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
print lr.score(X_test, y_test)









data = []



input = open('bw_freq.tsv')
for line in input:
    row = line.split('\t')
    x = row[0]
    size = float(row[1])
    data.append((x,size))

data.sort(key= lambda e: -int(e[1]))
print data
print '='*80
x,y = zip(*data)
plt.plot(y)

head_tail_breaks(data)


plt.show()

