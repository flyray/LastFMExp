import numpy as np
import theano.tensor as T
from theano import function

a = T.scalar('a')
f = function([a], [a**2, a**3])

result = f(3)
print result
print result[0]

