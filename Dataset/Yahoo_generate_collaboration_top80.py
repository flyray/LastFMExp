import sys
from sets import Set
import random


yahooData_address = sys.argv[1]
for dataDay in dataDays:
		fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay	+'.userID'
		with open(fileName, 'r') as f:
			# reading file line ie observations running one at a time
			for line in f:
				totalObservations +=1

				tim, article_chosen, click, user_features, userID = parseLine_userID(line)
				currentUser_featureVector = user_features[:-1]
				currentUserID = getIDAssignment(np.asarray(currentUser_featureVector), userFeatureVectors)                			
a = a[:80]

label = [0]
with open(sys.argv[2][:-11],'r') as fin:
	for line in fin:		
		label.append(int(line))


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

fin = open(sys.argv[2],'r')
cut_rd = [int(x) for x in fin.readline().strip('[]\n').split(',')]
cut_mx = [int(x) for x in fin.readline().strip('[]\n').split(',')]

#num = sys.argv[2].split('.')[-2]



#generate random arm_pool and write to file
fout_Part1 = open(sys.argv[1].split('/')[0]+'/processed_events_shuffled_'+'top100'+'_rand_part1.dat','w')
fout_Part1.write('userid	timestamp	arm_pool\n')
fout_Part2 = open(sys.argv[1].split('/')[0]+'/processed_events_shuffled_'+'top100'+'_rand_part2.dat','w')
fout_Part2.write('userid	timestamp	arm_pool\n')
for t in user_arm_tag:	
	if cut_rd[a.index(label[t['uid']])] == 1:
		#print t['aid']
		random_pool_1 = [t['aid']]+random.sample(user_arm_pool[t['uid']], 24)
		#print random_pool_1
		fout_Part1.write(str(t['uid'])+'\t'+str(t['tstamp'])+'\t'+str(random_pool_1)+'\n')		
	else:
		random_pool_2 = [t['aid']]+random.sample(user_arm_pool[t['uid']], 24)
		fout_Part2.write(str(t['uid'])+'\t'+str(t['tstamp'])+'\t'+str(random_pool_2)+'\n')
	
fout_Part1.close()
fout_Part2.close()
	
fout_Part1 = open(sys.argv[1].split('/')[0]+'/processed_events_shuffled_'+'top100'+'_max_part1.dat','w')
fout_Part1.write('userid	timestamp	arm_pool\n')
fout_Part2 = open(sys.argv[1].split('/')[0]+'/processed_events_shuffled_'+'top100'+'_max_part2.dat','w')
fout_Part2.write('userid	timestamp	arm_pool\n')
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


