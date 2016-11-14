import sys
from sets import Set
import random

user_arm_tag = []
# remove duplicate events
fin = open(sys.argv[1], 'r')
part1Num = int(sys.argv[2])
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

# filter arm pool for each user
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
# generate random arm_pool and write to file
fout_Part1 = open(sys.argv[1].split('/')[1] + '/processed_events_shuffled_Part1_' + str(part1Num) + '.dat', 'w')
fout_Part1.write('userid	timestamp	arm_pool\n')
fout_Part2 = open(sys.argv[1].split('/')[1] + '/processed_events_shuffled_Part2_' + str(part1Num) + '.dat', 'w')
fout_Part2.write('userid	timestamp	arm_pool\n')
for t in user_arm_tag:
    if t['uid'] < part1Num:
        # print t['aid']
        random_pool_1 = [t['aid']] + random.sample(user_arm_pool[t['uid']], 24)
        # print random_pool_1
        fout_Part1.write(str(t['uid']) + '\t' + str(t['tstamp']) + '\t' + str(random_pool_1) + '\n')
    else:
        random_pool_2 = [t['aid']] + random.sample(user_arm_pool[t['uid']], 24)
        fout_Part2.write(str(t['uid']) + '\t' + str(t['tstamp']) + '\t' + str(random_pool_2) + '\n')

fout_Part1.close()
fout_Part2.close()
