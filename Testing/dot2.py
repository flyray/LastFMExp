__author__ = 'Dianlei Zhang'
import numpy as np

theta = (
     [[2, 8, 9],
     [4, 5, 1],
     [7, 3, 6]]
)

theta1 = [2, 8, 9]
theta2 = [4, 5, 1]
theta3 = [7, 3, 6]

theta0 = []
theta0.append(theta1)
theta0.append(theta2)
theta0.append(theta3)

b = [[1],
     [2],
     [3]]


# c = np.dot(a, b)

aa = [3, 2, 9]
bb = [2, 6, 4]
cc = [7, 4, 3]

X = ([[3, 2, 9], [2, 6, 4], [7, 4, 3]])

# ccc = np.dot(aa, bb)
# cc = np.dot(aa, a)
c = np.dot(aa, bb)
# np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
var_aa = np.sqrt(np.dot(np.dot(aa, theta0), aa))
var_bb = np.sqrt(np.dot(np.dot(bb, theta0), bb))
var_cc = np.sqrt(np.dot(np.dot(cc, theta0), cc))

var_a = np.sqrt(np.dot(np.dot(X, theta0), np.transpose(X)))

print c
# print cc

print 'var_aa:', var_aa
print 'var_bb:', var_bb
print 'var_cc:', var_cc
print 'var_a:', var_a

print 'np.dot(aa, theta):', np.dot(aa, theta)
print 'np.dot(bb, theta):', np.dot(bb, theta)
print 'np.dot(cc, theta):', np.dot(cc, theta)
print 'np.dot(X, theta):', np.dot(X, theta)

print 'np.dot(np.dot(aa, theta), aa):', np.dot(np.dot(aa, theta), aa)
print 'np.dot(np.dot(bb, theta), bb):', np.dot(np.dot(bb, theta), bb)
print 'np.dot(np.dot(cc, theta), cc):', np.dot(np.dot(cc, theta), cc)
print 'np.dot(np.dot(, theta), np.transpose(X)):', np.sqrt(np.diag(np.dot(np.dot(X, theta), np.transpose(X))))



