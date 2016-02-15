import random
import time
import matplotlib.pyplot as plt
import heapq


def partition(vector, left, right, pivotIndex):
    pivotValue = vector[pivotIndex]
    vector[pivotIndex], vector[right] = vector[right], vector[pivotIndex]  # Move pivot to end
    storeIndex = left
    for i in range(left, right):
	vi = vector[i]
        if vi < pivotValue:
            vector[storeIndex], vi = vi, vector[storeIndex]
            storeIndex += 1
    vector[right], vector[storeIndex] = vector[storeIndex], vector[right]  # Move pivot to its final place
    return storeIndex
 
def _select(vector, left, right, k):
    "Returns the k-th smallest, (k >= 0), element of vector within vector[left:right+1] inclusive."
    while True:
        pivotIndex = random.randint(left, right)     # select pivotIndex between left and right
        pivotNewIndex = partition(vector, left, right, pivotIndex)
        pivotDist = pivotNewIndex - left
        if pivotDist == k:
            return vector[pivotNewIndex]
        elif k < pivotDist:
            right = pivotNewIndex - 1
        else:
            k -= pivotDist + 1
            left = pivotNewIndex + 1
 
def select(vector, k, left=None, right=None):
    if left is None:
        left = 0
    lv1 = len(vector) - 1
    if right is None:
        right = lv1
    return _select(vector, left, right, k)

def selectKth(arr, k):
	fro = 0
	to = len(arr)-1	 
	#if from == to we reached the kth element
	while (fro < to):
		r = fro
		w = to
		mid = arr[(r + w) / 2]
	 
		#stop if the reader and writer meets
		while (r < w):
			if (arr[r] >= mid): #put the large values at the end
				tmp = arr[w]
	    			arr[w] = arr[r]
	    			arr[r] = tmp
		    		w-=1
	 		else: #the value is smaller than the pivot, skip
	    			r+=1;
		#if we stepped up (r++) we need to step one down
	  	if (arr[r] > mid):
	   		r-=1
	 
		#the r pointer is on the end of the first k elements
	  	if (k <= r):
	   		to = r
	  	else:
	  		fro = r + 1
	 
	return arr[k]
 
if __name__ == '__main__':
    ts1=[]
    ts2 = []
    ts3 = []
    ns = xrange(1000, 10000, 1000)
    k = 5
    for n in ns:
	
	t_sum = 0
	for _ in range(0,100):
	    	v = []
		for _ in range(0,n):
			v.append(random.randint(0,1000))
	    	start = time.time()
		select(v, k)
	    	end = time.time()
		t_sum+=end-start
	
	ts1.append(t_sum/100.)

	v = []
	for _ in range(0,n):
		v.append(random.randint(0,10))

    	start = time.time()
        heapq.nlargest(k, v)
    	end = time.time()
        ts2.append(end-start)

    	v = []
	for _ in range(0,n):
		v.append(random.randint(0,10))

    	start = time.time()
        v.sort()
    	end = time.time()
        ts3.append(end-start)


	print n
    plt.scatter(ns, ts1, color='blue', label='quick select')
    plt.scatter(ns, ts2, color='red', label='heap')
    plt.scatter(ns, ts3, color='green', label='sort')
    plt.legend()
    plt.show()
    print ns
    print ts1
    print ts2
    print ts3
