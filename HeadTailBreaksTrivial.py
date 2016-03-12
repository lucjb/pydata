import matplotlib.pyplot as plt
import numpy as np

def mean(x):
	return sum(x)/len(x)

def head_tail_breaks(data, breaks=[]):
	n = len(data)
	m = mean(data)
	head = [x for x in data if x>m]
	tail = [x for x in data if x<=m]
	breaks.append(m)
	if len(head) < n*0.5 and len(head)>0:
		return head_tail_breaks(head, breaks)
	else:
		return breaks	


data = [x**-2 for x in range(1,20)]
breaks = head_tail_breaks(data)
plt.yticks(breaks)
breaks.sort()
plt.plot(data)
plt.show()												

breaks.append(min(data))
breaks.append(max(data))
breaks.sort()
f, _ = np.histogram(data, bins=breaks)
plt.hist(data, bins=breaks, orientation='horizontal')
plt.show()

