import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

n_samples = len(X_digits)

X_train = X_digits[:.9 * n_samples]
y_train = y_digits[:.9 * n_samples]
X_test = X_digits[.9 * n_samples:]
y_test = y_digits[.9 * n_samples:]

logistic = linear_model.LogisticRegression()
logistic.fit(X_train, y_train)
print('LogisticRegression score: %f' % logistic.score(X_test, y_test))


ind_probs = np.zeros((64, 17))
total = len(X_train)
for x in X_train:
	for i,v in enumerate(x):
		ind_probs[i,v]+=1./total


for x in X_test:
	for i in range(0,1):
		x[random.randint(0,63)]=-1	
print('LogisticRegression score: %f' % logistic.score(X_test, y_test))

test_preds = []
for x in X_test:
	lost_bit = np.where(x==-1)[0][0]
	x[lost_bit]=np.argmax(ind_probs[lost_bit])
	test_preds.append(logistic.predict(x))
	x[lost_bit]=-1
print accuracy_score(y_test, test_preds)			


test_preds = []
for x in X_test:
	lost_bit = np.where(x==-1)[0][0]
	probas = np.zeros((1,10))
	for v in range(0,17):
		x[lost_bit]=v
		probas+=logistic.predict_proba(x)*ind_probs[lost_bit, v]
	test_preds.append(np.argmax(probas))
print accuracy_score(y_test, test_preds)			

