from YahooExp_util_functions import *
import sys
import random
import numpy as np

def initializeW_top80(userFeatureVectors, sparsityLevel):
    W = np.zeros(shape = (80, 80))    
    for i in range(80):
            sSim = 0
            for j in range(80):
                sim = np.dot(userFeatureVectors[a[i]], userFeatureVectors[a[j]])
                W[i][j] = sim
                sSim += sim            
            W[i] /= sSim
    SparseW = W
    return SparseW.T

usernum = [(0, i) for i in range(160)]

yahooData_address = sys.argv[1]
dataDays = ['01']
for dataDay in dataDays:
		fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay	+'.userID.statistic'
		with open(fileName, 'r') as f:
			# reading file line ie observations running one at a time
			for i, line in enumerate(f):
				usernum[i] = (usernum[i][0]+int(line), i)

usernum.sort()
usernum.reverse()
a = [x[1] for x in usernum]
a = a[:80]
n = 80

userFeatureVectors = getClusters('YahooKMeansModel/10kmeans_model160.dat')	
W = initializeW_top80(userFeatureVectors, 160)
print W

weight_sum = np.sum(W)
print weight_sum
# random algorithm
iter_num = 10000
max_weight = 0
p = n*[0]
for iterate in range(iter_num):
	# random assign
	for j in range(n):
		p[j] = random.randint(0,1)	
	rand = p[:]
	org_weight = 0
	for i in range(n):
		for j in range(n):
			if p[i]!=p[j]:
				org_weight += W[i][j]

	

	# hill climbing
	for i in range(n):
		t = 0		
		for j in range(n):
			if p[i]!=p[j]:
				t += W[i][j]
		if (t < np.sum(W[i])-t):
			p[i] = 1-p[i]
	now_weight = 0
	# get result			
	for i in range(n):
		for j in range(n):
			if p[i]!=p[j]:
				now_weight += W[i][j]
	if now_weight > max_weight:
		max_iter = iterate
		max_weight = now_weight
		result = p[:]
		fout = open(yahooData_address+'/Yahoo.top80.cut','w')
		fout.write(str(rand)+'\n')
		fout.write(str(result)+str(len([x for x in p if x==0]))+'\n')
		print max_iter, org_weight, max_weight


