import sys
from sets import Set
import random
import numpy as np
#lastfm
a = [6, 131, 139, 122, 106, 121, 41, 2, 96, 22, 180, 46, 169, 167, 143, 71, 126, 82, 36, 31, 23, 171, 144, 134, 127, 50, 47, 44, 7, 195, 151, 102, 52, 174, 173, 168, 155, 137, 132, 125, 117, 108, 88, 59, 55, 25, 24, 13, 9, 199, 163, 146, 145, 130, 115, 86, 67, 58, 27, 5, 189, 183, 175, 165, 161, 158, 156, 135, 124, 107, 99, 95, 60, 48, 43, 42, 191, 187, 186, 172, 162, 157, 128, 120, 114, 110, 109, 100, 98, 92, 91, 90, 83, 54, 53, 51, 49, 45, 39, 37, 30, 29, 19, 18, 16, 15, 14, 3, 197, 192, 190, 185, 170, 166, 152, 147, 140, 129, 119, 116, 93, 87, 85, 79, 74, 69, 34, 33, 32, 28, 21, 8, 198, 182, 138, 133, 123, 103, 101, 94, 89, 80, 78, 72, 68, 61, 56, 40, 17, 12, 11, 10, 4, 0, 196, 179, 178, 164, 159, 154, 150, 141, 113, 111, 105, 104, 97, 81, 77, 76, 73, 66, 57, 38, 35, 26, 20, 194, 188, 184,160, 148, 142, 84, 62, 1, 193, 181, 177, 149, 136, 65, 63, 176, 153, 118, 75, 70, 64, 112]
#delicious
#a=[24, 18, 121, 112, 99, 96, 26, 124, 97, 27, 118, 114, 61, 46, 34, 19, 192, 122, 111, 55, 167, 160, 134, 86, 81, 57, 17, 10, 197, 176, 158, 133, 126, 113, 105, 93, 91, 87, 74, 64, 52, 39, 36, 16, 11, 9, 195, 193, 174, 168, 163, 150, 145, 141, 123, 117, 108, 107, 103, 79, 68, 33, 22, 186, 183, 180, 172, 156, 154, 143, 136, 129, 100, 90, 83, 77, 70, 65, 51, 49, 25, 20, 15, 14, 4, 199, 189, 188, 187, 185, 184, 182, 181, 179, 178, 169, 165, 164, 149, 147, 130, 115, 110, 104, 95, 76, 59, 44, 28, 23, 8, 6, 5, 198, 190, 177, 175, 171, 170, 166, 155, 153, 148, 144, 140, 127, 106, 88, 78, 58, 48, 38, 30, 12, 1, 196, 194, 191, 173, 161, 159, 157, 137, 135, 131, 120, 119, 102, 101, 85, 84, 73, 72, 50, 43, 35, 162, 152, 151, 142, 138, 92, 82, 80, 66, 60, 45, 41, 37, 21, 7, 2, 139, 128, 98, 94, 89, 71, 69, 56, 53, 42, 40, 32, 31, 29, 3, 132, 125, 109, 75, 67, 63, 62, 54, 47, 0, 146, 116, 13]
#a = a[:100]
print len(a)

label = [0]
with open(sys.argv[2],'r') as fin:
	for line in fin:		
		label.append(int(line))

n = 200
W = np.zeros([n, n])
with open(sys.argv[3]) as f:
    for line in f:
        line = line.split('\t')
        if line[0] != 'userID' and label[int(line[0])] in a and label[int(line[1])] in a:
            W[a.index(label[int(line[0])])][a.index(label[int(line[1])])] += 1 
for i in range(n):
    W[i][i] = 0


p = a[:50]
sum_connection = 0
for i in range (50):
	max_connection = 0
	max_x = 0
	for j in range(100,200):
		x = a[j]
		if not x in p:			
			connection = 0
			for k in range(50):
				connection += W[k][j]+W[j][k]
			if max_connection < connection:
				max_connection = connection
				max_x = x
	p.append(max_x)
	sum_connection += max_connection
print p
print sum_connection

user_arm_tag = []
#remove duplicate events
fin = open(sys.argv[1], 'r')
fin.readline()
last = {}
for line in fin:	
	arr = line.strip().split('\t')
	t = {}
	t['uid'] = int(arr[0])
	t['aid'] = int(arr[1])	
	t['tstamp'] = int(arr[3])
	if not t == last and label[int(t['uid'])] in a:
		last = t
		user_arm_tag.append(t)

print 'event number: '+str(len(user_arm_tag))

#filter arm pool for each user
user_arm_pool = {}
arm_pool = Set([])
for t in user_arm_tag:
	arm_pool.add(t['aid'])

for t in user_arm_tag:
	if not (t['uid'] in user_arm_pool):
		user_arm_pool[t['uid']] = arm_pool.copy()		
	if t['aid'] in user_arm_pool[t['uid']]:
		user_arm_pool[t['uid']].remove(t['aid'])	
random.shuffle(user_arm_tag)

	
fout_Part1 = open(sys.argv[1].split('/')[0]+'/processed_events_shuffled_'+'select100'+'_max_observation_part1.dat','w')
fout_Part1.write('userid	timestamp	arm_pool\n')
fout_Part2 = open(sys.argv[1].split('/')[0]+'/processed_events_shuffled_'+'select100'+'_max_oberservation_part2.dat','w')
fout_Part2.write('userid	timestamp	arm_pool\n')
for t in user_arm_tag:	
	if label[t['uid']] in p:
		if p.index(label[t['uid']]) < 50:
			#print label[t['uid']]
			#print t['aid']
			random_pool_1 = [t['aid']]+random.sample(user_arm_pool[t['uid']], 24)
			#print random_pool_1
			fout_Part1.write(str(t['uid'])+'\t'+str(t['tstamp'])+'\t'+str(random_pool_1)+'\n')		
		else:
			random_pool_2 = [t['aid']]+random.sample(user_arm_pool[t['uid']], 24)
			fout_Part2.write(str(t['uid'])+'\t'+str(t['tstamp'])+'\t'+str(random_pool_2)+'\n')
		
fout_Part1.close()
fout_Part2.close()


