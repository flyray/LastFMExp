import sys
import random
import operator
import matplotlib.pyplot as plt
from sets import Set
# get cluster assignment of V, M is cluster centroids
def getIDAssignment(V, M):
        MinDis = float('+inf')
        assignment = None
        for i in range(M.shape[0]):
            dis = distance.euclidean(V, M[i])
            if dis < MinDis:
                assignment = i
                MinDis = dis
        return assignment

# This code simply reads one line from the source files of Yahoo!

def parseLine(line):
        line = line.split("|")
        
        tim, articleID, click = line[0].strip().split(" ")
        tim, articleID, click = int(tim), int(articleID), int(click)
        user_features = np.array([float(x.strip().split(':')[1]) for x in line[1].strip().split(' ')[1:]])
        
        pool_articles = [l.strip().split(" ") for l in line[2:]]
        pool_articles = np.array([[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
        return tim, articleID, click, user_features, pool_articles

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
	if not t == last:
		last = t
		user_arm_tag.append(t)

print 'event number: '+str(len(user_arm_tag))


article_stat = {}
for t in user_arm_tag:
	if not t['aid'] in article_stat:
		article_stat[t['aid']] = 1
	else:
		article_stat[t['aid']] += 1

sorted_article = sorted(article_stat.items(), key=operator.itemgetter(1))

frac = 0.2
sample = random.sample(sorted_article, int(frac*len(sorted_article)))
sample_set = Set([x[0] for x in sample])

print len(sample_set)

y = sample
plt.plot(range(len(y)), [x[1] for x in sorted(y, key=operator.itemgetter(1))])
plt.show()

fout = open(sys.argv[2]+'.sample20','w')
fin = open(sys.argv[2], 'r')
fin.readline()
for line in fin:
    userID, tim, pool_articles = parseLine(line)
    if int(pool_articles[0]) in sample_set:
        fout.write(line)

