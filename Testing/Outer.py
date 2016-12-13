__author__ = 'Dianlei Zhang'
import numpy as np

A = {}
article = [1, 2, 3, 4]
article = np.asarray(article)

A = np.outer(article, article)

b = [1, 2, 3, 4]
b = np.asarray(b)

# theta = np.dot(np.linalg.inv(A), b)

print A
# print theta
