import sys
from sets import Set
import random

user_arm_tag = []

# remove duplicate events
fin = open(sys.argv[1], 'r')
fin.readline()
last = {}
for line in fin:
    arr = line.strip().split('\t')
    t = {}
    t['uid'] = int(arr[0])
    t['aid'] = int(arr[1])
    t['tstamp'] = int(arr[3])
    if not t == last:
        last = t
        user_arm_tag.append(t)
print 'event number: ' + str(len(user_arm_tag))

# get all arms
arm_pool = Set([])
for t in user_arm_tag:
    arm_pool.add(t['aid'])

# generate random arm_pool and write to file
fout = open(sys.argv[1].split('/')[0] + '/processed_events.dat', 'w')
fout.write('userid	timestamp	arm_pool')
for t in user_arm_tag:
    random_pool = [t['aid']] + random.sample(arm_pool, 24)
    fout.write(str(t['uid']) + '\t' + str(t['tstamp']) + '\t' + str(random_pool) + '\n')
fout.close()
