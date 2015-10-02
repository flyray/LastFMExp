import sys
import random
import numpy as np

labelfile = sys.argv[1]
relationFileName = sys.argv[2]

# read cluster label
label = [0]
with open(labelfile,'r') as fin:
	for line in fin:		
	    label.append(int(line))
label = np.array(label)

# create cluster adj matrix
n = int(labelfile.split('.')[-1])

W = np.zeros([n, n])
with open(relationFileName) as f:
    for line in f:
        line = line.split('\t')
        if line[0] != 'userID':                   
            W[label[int(line[0])]][label[int(line[1])]] += 1 
for i in range(n):
    W[i][i] = 0

weight_sum = np.sum(W)/2
print weight_sum
# random algorithm
iter_num = 10000
max_weight = 0
p = n*[0]
for iterate in range(iter_num):
	# random assign
	for j in range(1, n):
		p[j] = random.randint(0,1)	
	rand = p[:]
	org_weight = 0
	for i in range(1, n):
		for j in range(i, n):
			if p[i]!=p[j]:
				org_weight += W[i][j]

	

	# hill climbing
	for i in range(1, n):
		t = 0		
		for j in range(1, n):
			if p[i]!=p[j]:
				t += W[i][j]
		if (t < np.sum(W[i])-t):
			p[i] = 1-p[i]
	now_weight = 0
	# get result			
	for i in range(1, n):
		for j in range(i, n):
			if p[i]!=p[j]:
				now_weight += W[i][j]
	if now_weight > max_weight:
		max_iter = iterate
		max_weight = now_weight
		result = p[:]
		fout = open(labelfile+'.cut','w')
		fout.write(str(rand)+'\n')
		fout.write(str(result)+'\n')
		print max_iter, org_weight, max_weight


