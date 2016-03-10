import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
import itertools
from sklearn import cross_validation
from sklearn.utils import shuffle

amounts_of_missing_features = xrange(1,5,1)
times = []
full_accs = []
naive_accs = []
maxp_accs = []
ctpl_accs =[]
avgs_accs = []
pixels = 8*8
levels = 17
n=2000

for d in amounts_of_missing_features:
	print d
	digits = datasets.load_digits()
	#digits = fetch_mldata("MNIST original")
	digits.data = digits.data.astype(int)
    	digits.data, digits.target = shuffle(digits.data, digits.target)
	X = digits.data[:n]
	y = digits.target[:n]
#	plt.matshow(X[0].reshape((8,8)), cmap=plt.cm.gray_r)
#	plt.show()

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

	logistic = linear_model.LogisticRegression()
	logistic.fit(X_train, y_train)
	full_acc = logistic.score(X_test, y_test)
	print('Full feature set accuracy: %f' % full_acc)
	full_accs.append(full_acc)

	# Compute independent probabilities of all possible feature values
	ind_probs = np.zeros((pixels, levels))
	total = len(X_train)
	for x in X_train:
		for i,v in enumerate(x):
			ind_probs[i,v]+=1./total
	
	#Compute average pixel values
	avgs = np.zeros(pixels)
	for x in X_train:
		for i,v in enumerate(x):
			avgs[i]+=v
	avgs = avgs/total

	# Now we randomly set d pixels to -1, which means the value is missing. This value has never been seen during training.
	for x in X_test:
		for i in range(0,d):
			r = random.randint(0,pixels-1)
			x[r]=-1
	#plt.matshow(x.reshape((8,8)))
	#plt.show()
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

	# Baseline: assign to each missing feature the most likely value
	test_preds = []
	for x in X_test:
		lost_bits = np.where(x==-1)[0]
		for lost_bit in lost_bits:
			x[lost_bit]=avgs[lost_bit]
		test_preds.append(logistic.predict(x))
		x[lost_bits]=-1
	avgs_acc = accuracy_score(y_test, test_preds)
	avgs_accs.append(avgs_acc)
	print 'Average fill in accuracy: %f' % avgs_acc			
		

	# Trick: Compute the probability of each class conditioned on all features using the conditional total probability law.
	test_preds = []
	valid_values = range(0,levels)
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
		if (ii % 10)==0:
			print ii, nt, ii/float(nt)
	ctpl_acc = accuracy_score(y_test, test_preds)
	print 'Conditional Total Probability Law computation accuracy: %f' % ctpl_acc
	ctpl_accs.append(ctpl_acc)

plt.plot(amounts_of_missing_features, full_accs, color='black', label='No missing features')
plt.plot(amounts_of_missing_features, naive_accs, color='blue', label='Do nothing')
plt.plot(amounts_of_missing_features, maxp_accs, color='red', label='Most likely value')
plt.plot(amounts_of_missing_features, avgs_accs, color='cyan', label='Average value')
plt.plot(amounts_of_missing_features, ctpl_accs, color='green', label='CTPL')
plt.ylabel('Classification Accuracy')
plt.xlabel('Amount of missing features')
plt.title('Classification Accuracy vs Amount of missing features')
plt.legend(loc='lower left')
plt.show()
									
