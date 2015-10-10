import sys
import random
import numpy as np
#lastfm
#a = [6, 131, 139, 122, 106, 121, 41, 2, 96, 22, 180, 46, 169, 167, 143, 71, 126, 82, 36, 31, 23, 171, 144, 134, 127, 50, 47, 44, 7, 195, 151, 102, 52, 174, 173, 168
#, 155, 137, 132, 125, 117, 108, 88, 59, 55, 25, 24, 13, 9, 199, 163, 146, 145, 130, 115, 86, 67, 58, 27, 5, 189, 183, 175, 165, 161, 158, 156, 135, 124, 107, 99, 95, 60, 48, 43, 42, 191, 187, 186, 172, 162, 157, 128, 120, 114, 110, 109, 100
#, 98, 92, 91, 90, 83, 54, 53, 51, 49, 45, 39, 37, 30, 29, 19, 18, 16, 15, 14, 3, 197, 192, 190, 185, 170, 166, 152, 147, 140, 129, 119, 116, 93, 87, 85, 79, 74,
# 69, 34, 33, 32, 28, 21, 8, 198, 182, 138, 133, 123, 103, 101, 94, 89, 80, 78, 72, 68, 61, 56, 40, 17, 12, 11, 10, 4, 0, 196, 179, 178, 164, 159, 154, 150, 141,
# 113, 111, 105, 104, 97, 81, 77, 76, 73, 66, 57, 38, 35, 26, 20, 194, 188, 184,160, 148, 142, 84, 62, 1, 193, 181, 177, 149, 136, 65, 63, 176, 153, 118, 75, 70, 64, 112]
a=[24, 18, 121, 112, 99, 96, 26, 124, 97, 27, 118, 114, 61, 46, 34, 19, 192, 122, 111, 55, 167, 160, 134, 86, 81, 57, 17, 10, 197, 176, 158, 133, 126, 113, 105, 93, 91, 87, 74, 64, 52, 39, 36, 16, 11, 9, 195, 193, 174, 168, 163, 150, 145, 141, 123, 117, 108, 107, 103, 79, 68, 33, 22, 186, 183, 180, 172, 156, 154, 143, 136, 129, 100, 90, 83, 77, 70, 65, 51, 49, 25, 20, 15, 14, 4, 199, 189, 188, 187, 185, 184, 182, 181, 179, 178, 169, 165, 164, 149, 147, 130, 115, 110, 104, 95, 76, 59, 44, 28, 23, 8, 6, 5, 198, 190, 177, 175, 171, 170, 166, 155, 153, 148, 144, 140, 127, 106, 88, 78, 58, 48, 38, 30, 12, 1, 196, 194, 191, 173, 161, 159, 157, 137, 135, 131, 120, 119, 102, 101, 85, 84, 73, 72, 50, 43, 35, 162, 152, 151, 142, 138, 92, 82, 80, 66, 60, 45, 41, 37, 21, 7, 2, 139, 128, 98, 94, 89, 71, 69, 56, 53, 42, 40, 32, 31, 29, 3, 132, 125, 109, 75, 67, 63, 62, 54, 47, 0, 146, 116, 13]
a = a[:100]
labelfile = sys.argv[1]
relationFileName = sys.argv[2]

# read cluster label
label = [0]
with open(labelfile,'r') as fin:
	for line in fin:		
	    label.append(int(line))
label = np.array(label)

# create cluster adj matrix
#n = int(labelfile.split('.')[-1])
n = len(a)
print a
W = np.zeros([n, n])
with open(relationFileName) as f:
    for line in f:
        line = line.split('\t')
        if line[0] != 'userID' and label[int(line[0])] in a and label[int(line[1])] in a:
            W[a.index(label[int(line[0])])][a.index(label[int(line[1])])] += 1 
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
		fout = open(labelfile+'.top100.cut','w')
		fout.write(str(rand)+'\n')
		fout.write(str(result)+str(len([x for x in p if x==0]))+'\n')
		print max_iter, org_weight, max_weight


