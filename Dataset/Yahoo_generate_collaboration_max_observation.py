from YahooExp_util_functions import *
import sys
from sets import Set
import random

def initializeW_top(userFeatureVectors, sparsityLevel):
	n = 160
	W = np.zeros(shape = (n, n))    
	for i in range(n):
		sSim = 0
		for j in range(n):
			sim = np.dot(userFeatureVectors[a[i]], userFeatureVectors[a[j]])
			W[i][j] = sim
			sSim += sim            
		W[i] /= sSim
	SparseW = W
	return SparseW.T

usernum = [(0, i) for i in range(160)]

yahooData_address = sys.argv[1]
dataDays = ['01']#, '02', '03', '04', '05', '06', '07', '08', '09', '10']
for dataDay in dataDays:
	fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay	+'.userID.statistic'
	with open(fileName, 'r') as f:
		for i, line in enumerate(f):
			usernum[i] = (usernum[i][0]+int(line), i)

usernum.sort()
usernum.reverse()
a = [x[1] for x in usernum]
print a
userFeatureVectors = getClusters('YahooKMeansModel/10kmeans_model160.dat')	
W = initializeW_top(userFeatureVectors, 160)
print W
p = a[:40]
for i in range (40):
	max_connection = 0
	max_x = 0
	for j in range(80,160):
		x = a[j]
		if not x in p:			
			connection = 0
			for k in range(40):
				connection += W[k][j]+W[j][k]
			if max_connection < connection:
				max_connection = connection
				max_x = x
	p.append(max_x)
print p



with open('YahooKMeansModel/10kmeans_model160.dat', 'r') as fin:
	kmeans = fin.read().split('\n')
with open('YahooKMeansModel/10kmeans_model160.dat.80.max_observation', 'w') as fout:
	for x in p:
		fout.write(kmeans[x]+'\n')


dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
for dataDay in dataDays:
	fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay	+'.userID'

	fout_Part1 = open(fileName+'.max_observation.part1','w')
	fout_Part2 = open(fileName+'.max_observation.part2','w')

	with open (fileName, 'r') as fin:
		for line in fin:	
			line = line.split("|")
			userID = int(line[1])
			if userID in p:
				line[1] = str(a.index(userID))
				if p.index(userID) <40:
					fout_Part1.write('|'.join(line))
				else:
					fout_Part2.write('|'.join(line))
	fout_Part1.close()
	fout_Part2.close()


