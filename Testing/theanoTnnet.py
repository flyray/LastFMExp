# coding=utf-8
import numpy
import theano
import theano.tensor as T


inputMean = [1, 2, 3, 4]
inputBias = [0.5, 0.4, 0.3, 0.2]
Bias = [0.1, 0.4, 0.2, 0.1]
W = [
    [1, 3, 5, 6],
    [4, 3, 7, 2],
    [7, 8, 4, 3],
    [5, 9, 3, 2]
]

estimateReward = T.nnet.softmax(T.dot(inputMean, W) + inputBias + Bias)

result1 = estimateReward[0]
result2 = estimateReward[0, 1]
result3 = estimateReward[0, 3]
# print theano.pprint(estimateReward)

f = theano.function([], [result1])
f2 = theano.function([], [result2])
f3 = theano.function([], [result3])


print f()
print f2()
print f3()
