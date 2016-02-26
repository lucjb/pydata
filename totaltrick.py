import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
import itertools


amounts_of_missing_features = xrange(1,4,1)
times = []
full_accs = []
naive_accs = []
maxp_accs = []
ctpl_accs =[]

for d in amounts_of_missing_features:
	print d
	digits = datasets.load_digits()
	X_digits = digits.data
	y_digits = digits.target

	n_samples = len(X_digits)

	X_train = X_digits[:.7 * n_samples]
	y_train = y_digits[:.7 * n_samples]
	X_test = X_digits[.7 * n_samples:]
	y_test = y_digits[.7 * n_samples:]

	logistic = linear_model.LogisticRegression()
	logistic.fit(X_train, y_train)
	full_acc = logistic.score(X_test, y_test)
	print('Full feature set accuracy: %f' % full_acc)
	full_accs.append(full_acc)

	# Compute independent probabilities of all possible feature values
	ind_probs = np.zeros((64, 17))
	total = len(X_train)
	for x in X_train:
		for i,v in enumerate(x):
			ind_probs[i,v]+=1./total


	# Now we randomly set d pixels to -1, which means the value is missing. This value has never been seen in training.
	for x in X_test:
		for i in range(0,d):
			x[random.randint(0,63)]=-1	
	naive_acc = logistic.score(X_test, y_test)
	print('Naive plain prediction accuracy: %f' % naive_acc)
	naive_accs.append(naive_acc)


	# Baseline: assign to each missing feature the most likely value
	test_preds = []
	for x in X_test:
		lost_bits = np.where(x==-1)[0]
		for lost_bit in lost_bits:
			x[lost_bit]=np.argmax(ind_probs[lost_bit])
		test_preds.append(logistic.predict(x))
		x[lost_bits]=-1
	maxp_acc = accuracy_score(y_test, test_preds)
	maxp_accs.append(maxp_acc)
	print 'Most porbable fill in accuracy: %f' % maxp_acc			

	# Compute the probability of each class conditioned on all features using the conditional total probability law.
	test_preds = []
	valid_values = range(0,17)
	nt = len(X_test)
	for ii, x in enumerate(X_test): 
		lost_bits = np.where(x==-1)[0]
		permutations = itertools.product(valid_values, repeat=len(lost_bits))
		probas = np.zeros((1,10))
		for j, perm in enumerate(permutations):
			ind_p_prod = 1.	
			for i, v in enumerate(perm):
				lost_bit = lost_bits[i]
				x[lost_bit]=v
				ind_p_prod *= ind_probs[lost_bit,v]
			probas += logistic.predict_proba(x)*ind_p_prod
		test_preds.append(np.argmax(probas))
		print ii, nt, ii/float(nt)
	ctpl_acc = accuracy_score(y_test, test_preds)
	print 'Conditional Total Probability Law computation accuracy: %f' % ctpl_acc
	ctpl_accs.append(ctpl_acc)

plt.plot(amounts_of_missing_features, full_accs, color='black')
plt.plot(amounts_of_missing_features, naive_accs, color='blue')
plt.plot(amounts_of_missing_features, maxp_accs, color='red')
plt.plot(amounts_of_missing_features, ctpl_accs, color='green')
plt.show()
									
