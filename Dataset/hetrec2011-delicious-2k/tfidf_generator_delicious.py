import sklearn
import sys
import re
import numpy
import scipy.sparse
import scipy.io
from collections import defaultdict
from sets import Set
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

# pre-process events
# file = open(sys.argv[1]+'user_taggedbookmarks-timestamps.dat', 'r')
file = open('./' + 'user_taggedbookmarks-timestamps.dat', 'r')
file.readline()
raw_user_bookmark_tag = []
max = 1000000
for line in file:
    max -= 1
    if max < 0:
        break
    arr = line.strip().split('\t')
    t = {}
    t['uid'] = int(arr[0])
    t['aid'] = int(arr[1])
    t['tid'] = int(arr[2])
    t['tstamp'] = int(arr[3])
    raw_user_bookmark_tag.append(t)

print 'raw event number: ' + str(len(raw_user_bookmark_tag))

# remove tags occuring less than 10
# ftagremove = open(sys.argv[1]+'tag_remove','w')
ftagremove = open('./' + 'tag_remove', 'w')
tag_count = defaultdict(int)
tag_remove = Set([])
for t in raw_user_bookmark_tag:
    tag_count[t['tid']] += 1

for tid in tag_count:
    if tag_count[tid] < 10:
        ftagremove.write(str(tid) + '\n')
        tag_remove.add(tid)
print 'removed tags: ' + str(len(tag_remove))
# build vocabulary
vocab = {}
tag2word = {}
file = open('./' + 'tags.dat', 'r')
file.readline()
for line in file:
    arr = line.strip().split('\t')
    if not int(arr[0]) in tag_remove:
        wlist = re.split(r"[ _-]+", arr[1])
        if not (int(arr[0]) in tag2word):
            tag2word[int(arr[0])] = wlist
        for word in wlist:
            if not word in vocab:
                vocab[word] = len(vocab)
            # print arr[0]+' '+str(tag2word[int(arr[0])])

print 'vocabulary size: ' + str(len(vocab))

num = 0
ftmp = open('tag_removed_log.dat', 'w')
ftmp.write('Hehe\n')
logs = []
for i, t in enumerate(raw_user_bookmark_tag):
    if raw_user_bookmark_tag[i]['tid'] in tag_remove:
        num += 1
    else:
        logs.append(t)
        ftmp.write(str(t['uid']) + '\t' + str(t['aid']) + '\t' + str(t['tid']) + '\t' + str(t['tstamp']) + '\n')
ftmp.close()
print 'removed event: ' + str(num)

# build index & document
bookmark2idx = {}
idx2bookmark = []
for i, t in enumerate(logs):
    if not t['aid'] in bookmark2idx:
        bookmark2idx[t['aid']] = len(bookmark2idx)
        idx2bookmark.append(t['aid'])
    # print str(bookmark2idx[t['aid']])+' '+str(len(idx2bookmark))

print bookmark2idx[76774]

# counting frequency
# x = numpy.zeros(shape = (len(bookmark2idx), len(vocab)))
count = scipy.sparse.lil_matrix((len(bookmark2idx), len(vocab)))
for t in logs:
    # print t
    aidx = bookmark2idx[t['aid']]
    # if t['aid'] == 1:
    #	print tag2word[t['tid']]
    for word in tag2word[t['tid']]:
        count[aidx, vocab[word]] += 1

# get tfidf
tfidf_transformer = TfidfTransformer()
x = tfidf_transformer.fit_transform(count)
print 'tfidf size: ' + str(x.shape)

svd = TruncatedSVD(
    n_components=25)  # Use TruncatedSVD instead of PCA or RandomizedPCA since the matrix is large and sparse.
result = svd.fit(x).transform(x)
print 'feature matrix size: ' + str(result.shape)

# fArticleFeatureVectors= open('./' + 'Arm_FeatureVectors.dat', 'w')
with open('./' + 'Arm_FeatureVectors.dat', 'a+') as f:
    f.write('ArticleID')
    f.write('\t')
    f.write('FeatureVector')
    f.write('\n')
for i, idx in enumerate(idx2bookmark):
    articleID = idx
    featureVector = result[i]
    with open('./' + 'Arm_FeatureVectors.dat', 'a+') as f:
        f.write(str(articleID))
        f.write('\t')
        f.write(str(featureVector))
        f.write('\n')

# print to file
fvectors = open('./' + 'feature_vectors.dat', 'w')
scipy.io.mmwrite(fvectors, result)  # To read the matrix back, use scipy.io.mmread()
fvectors.close()
fartistidx = open('./' + 'arm_idx.dat', 'w')
fartistidx.write("index" + '\t' + "armId" + '\n')
for i, idx in enumerate(idx2bookmark):
    fartistidx.write(str(i) + '\t' + str(idx) + '\n')
fartistidx.close()
