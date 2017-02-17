import numpy as np
import theano.tensor as T
from theano import function

a = 1
b = 2
if T.lt(a, b):
    print 'ok'
else:
    print 'no'
