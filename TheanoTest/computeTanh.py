import theano
import theano.tensor as T
import numpy as np

# defining the tensor variables
X = T.matrix("")
W = T.matrix("W")
b_sym = T.vector("b_sym")

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym), sequences=X)
compute_elementwise = theano.function(inputs=[X, W, b_sym], outputs=results)

# test values
x = np.eye(2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
b[1] = 2
x[0][1] = 3
w[0][1] = 2
w[1][1] = 3
print 'x: \n', x
print 'w: \n', w
print 'b: \n', b


print(compute_elementwise(x, w, b))

# comparison with numpy
print(np.tanh(x.dot(w) + b))

print 'x.dot(w): \n', x.dot(w)

