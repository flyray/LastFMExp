__author__ = 'Dianlei Zhang'

import numpy as np

a = [[1,2,3],
     [2,3,4],
     [3,4,5]]
b = np.identity(3)

A = np.kron(a, b)

print A