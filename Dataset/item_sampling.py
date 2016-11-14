import sys
import random
import operator
import matplotlib.pyplot as plt
from sets import Set


def parseLine(line):
    userID, tim, pool_articles = line.split("\t")
    userID, tim = int(userID), int(tim)
    pool_articles = pool_articles.strip('[').strip(']').strip('\n').split(',')
    return userID, tim, pool_articles


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

article_stat = {}
for t in user_arm_tag:
    if not t['aid'] in article_stat:
        article_stat[t['aid']] = 1
    else:
        article_stat[t['aid']] += 1

sorted_article = sorted(article_stat.items(), key=operator.itemgetter(1))

frac = 0.2
sample = random.sample(sorted_article, int(frac * len(sorted_article)))
sample_set = Set([x[0] for x in sample])

print len(sample_set)

y = sample
plt.plot(range(len(y)), [x[1] for x in sorted(y, key=operator.itemgetter(1))])
plt.show()

fout = open(sys.argv[2] + '.sample20', 'w')
fin = open(sys.argv[2], 'r')
fin.readline()
for line in fin:
    userID, tim, pool_articles = parseLine(line)
    if int(pool_articles[0]) in sample_set:
        fout.write(line)
