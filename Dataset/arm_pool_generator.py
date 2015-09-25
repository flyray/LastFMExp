import sys
from sets import Set
import random

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
	#print t['tstamp']
	if not t == last:
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
#generate random arm_pool and write to file
fout = open(sys.argv[1].split('/')[0]+'/processed_events_shuffled_tag_removed.dat','w')
fout.write('userid	timestamp	arm_pool\n')
for t in user_arm_tag:	
	random_pool = [t['aid']]+random.sample(user_arm_pool[t['uid']], 24)
	fout.write(str(t['uid'])+'\t'+str(t['tstamp'])+'\t'+str(random_pool)+'\n')
fout.close()

