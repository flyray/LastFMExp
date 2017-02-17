# coding=utf-8
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

# 定义变量:
x = T.vector('x')
w = theano.shared(np.array([1,1]))
b = theano.shared(-1.5)

# 定义数学表达式:
z = T.dot(x,w)+b
a = ifelse(T.lt(z,0),0,1)

neuron = theano.function([x],a)

# 定义输入和权重
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

# 遍历所有输入并得到输出:
for i in range(len(inputs)):
    t = inputs[i]
    out = neuron(t)
    print 'The output for x1=%d | x2=%d is %d' % (t[0],t[1],out)