import numpy as np
import theano.tensor as T
from theano import function
from theano import pp

x = T.dscalar('x')
y = x**3
qy = T.grad(y, x)
f = function([x], qy)
print f(4)
print pp(qy)

