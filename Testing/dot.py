__author__ = 'Dianlei Zhang'
import numpy as np

a = ([[1, 2, 11],
     [2, 3, 4],
     [3, 4, 5]])

b = [[1],
     [2],
     [3]]


# c = np.dot(a, b)

aa = [1, 2, 10]
bb = [2, 3, 4]
cc = [3, 4, 5]


# ccc = np.dot(aa, bb)
cc = np.dot(aa, a)
c = np.dot(aa, bb)
# np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
var_aa = np.sqrt(np.dot(np.dot(aa, a), aa))
var_bb = np.sqrt(np.dot(np.dot(bb, a), bb))
var_cc = np.sqrt(np.dot(np.dot(cc, a), cc))

var_a = np.sqrt(np.dot(np.dot(a, a), np.transpose(a)))

print c
print cc

# print 'var_a:', var_a
print 'cc:', cc
