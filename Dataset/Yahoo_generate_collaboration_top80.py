import sys
from sets import Set
import random

usernum = [(0, i) for i in range(160)]

yahooData_address = sys.argv[1].split('/')[0]
dataDays = ['01']
for dataDay in dataDays:
		fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay	+'.userID.statistic'
		with open(fileName, 'r') as f:
			for i, line in enumerate(f):
				usernum[i] = (usernum[i][0]+int(line), i)

usernum.sort()
usernum.reverse()
a = [x[1] for x in usernum]
a = a[:80]

with open(sys.argv[2],'r') as fin:
	cut_rd = [int(x) for x in fin.readline().strip('[]\n').split(',')]
	cut_mx = [int(x) for x in fin.readline().strip('[]\n').split(',')]
	
fout_Part1 = open(sys.argv[1]+'.max.part1','w')
fout_Part1 = open(sys.argv[1]+'.max.part2','w')

for t in user_arm_tag:	
	if cut_mx[a.index(label[t['uid']])] == 1:
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


