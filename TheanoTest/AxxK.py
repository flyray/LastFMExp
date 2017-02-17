import theano
import theano.tensor as T
import numpy as np

k = T.iscalar('k')
A = T.vector('A')

outputs, updates = theano.scan(lambda result, A: result * A, non_sequences=A, outputs_info=T.ones_like(A), n_steps=k)
result = outputs[-1]
fn_Ak = theano.function([A, k], result, updates=updates)
print fn_Ak(range(10), 3)

