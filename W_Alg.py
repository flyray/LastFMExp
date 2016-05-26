import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize
from scipy.optimize import minimize
import math
from util_functions import vectorize, matrixize


class WknowThetaStruct:
	def __init__(self, featureDimension, eta_, userNum, theta):
		self.theta = theta
		self.userNum = userNum
		self.featureDimension = featureDimension

		self.eta_ = eta_

		self.A = eta_ * np.identity(n = userNum*userNum)
		self.b = np.zeros(userNum*userNum)
		self.AInv = np.linalg.inv(self.A)
		self.W = np.zeros(shape = (userNum, userNum))
		self.CoTheta = np.dot(self.theta, self.W)

		self.f1 = 0.0
		self.f2 = (eta_/2.0)*np.matrix.trace(np.dot(np.transpose(self.W), self.W))
		

		self.BigTheta = np.kron(np.identity(n=userNum), self.theta)
		self.CCA = np.identity(n = featureDimension*userNum)
	def updateParameters(self, articlePicked, click, userID):
		Xi_Matirx = np.zeros(shape = (self.featureDimension, self.userNum))
		Xi_Matirx.T[userID] = articlePicked.featureVector
		X = vectorize( np.dot(np.transpose(self.theta), Xi_Matirx))

		#self.W = matrixize(np.dot(self.AInv, self.b), self.userNum)
		#fun = lambda x = np.identity(self.userNum): self.f1 + (1/2.0)*(np.dot( np.dot(articlePicked.featureVector, self.theta)  ,x.T[userID]) - click)**2 + (self.eta_/2.0)*np.matrix.trace(np.dot(np.transpose(x), x))
		def fun(x):
			x = np.asarray(x)
			x = x.reshape(self.userNum, self.userNum)
			return self.f1 + (1/2.0)*(np.dot( np.dot(articlePicked.featureVector, self.theta)  ,x.T[userID]) - click)**2 + (self.eta_/2.0)*np.matrix.trace(np.dot(np.transpose(x), x))

		def fprime(x):
			x = np.asarray(x)
			return np.dot(self.A+ np.outer(X, X), x) -(self.b + click*X) 

		cons = ({'type' : 'ineq',
				'fun' : lambda  x: x },
				{'type' : 'ineq',
				'fun' : lambda x: 1-x},
				{'type': 'eq',
				 'fun': lambda x : np.sum(np.square(np.sum(x, axis = 1)-1)) # I'm not sure whether to sum in row or column, but this should do the work.
				}
				)
		'''
		res = minimize(fun, self.W, constraints = cons, method ='SLSQP', jac = fprime, options={'disp': False})
		self.W = res.x.reshape(self.userNum, self.userNum)
		#self.W = np.transpose(self.W)
		'''
		

		#print self.W[0]
		#Normalization
		#self.W = normalize(self.W, axis=0, norm='l1')
		self.A += np.outer(X, X)
		self.b += click * X
		self.AInv = np.linalg.inv(self.A)
		self.W = matrixize(np.dot(self.AInv, self.b), self.userNum)

		self.f1 +=(1/2.0)*(np.dot( np.dot(articlePicked.featureVector, self.theta)  ,self.W.T[userID]))**2
		self.CoTheta = np.dot(self.theta, self.W)
		self.CCA = np.dot(np.dot(self.BigTheta, self.AInv), np.transpose(self.BigTheta))

	def getProb(self, alpha, article, userID):
		TempFeatureM = np.zeros(shape =(len(article.featureVector), self.userNum))
		TempFeatureM.T[userID] = article.featureVector
		TempFeatureV = vectorize(TempFeatureM)

		mean = np.dot(self.CoTheta.T[userID], article.featureVector)
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		pta = mean + alpha*var
		#print np.shape(mean)
		return pta




class WStruct:
	def __init__(self, featureDimension, lambda_, eta_, userNum):	
		self.userNum = userNum
		# Basic stat in estimating Theta
		self.T_A = lambda_*np.identity(n = featureDimension*userNum)
		self.T_b = np.zeros(featureDimension*userNum)
		self.UserTheta = np.zeros(shape = (featureDimension, userNum))
		self.UserTheta = np.random.random((featureDimension, userNum))
		

		# Basic stat in estimating W
		self.W_A = eta_*np.identity(n = userNum*userNum)
		self.W_b = np.zeros(userNum*userNum)
		
		self.W_AInv = np.linalg.inv(self.W_A)
		#self.W = matrixize(np.dot(self.W_AInv, self.W_b), self.userNum)
		#self.W =  np.zeros(shape = (userNum, userNum))
		self.W = np.identity(n = userNum)
		#self.W = np.random.random((userNum, userNum))

		for i in range(self.W.shape[0]):
			self.W[i] /= sum(self.W[i]) 
		
		self.CoTheta = np.dot(self.UserTheta, self.W)

		self.BigW = np.kron(np.transpose(self.W), np.identity(n=featureDimension))
		self.CCA = np.identity(n = featureDimension*userNum)

		self.BigTheta = np.kron(np.identity(n=userNum) , self.UserTheta)
		self.W_CCA = np.identity(n = featureDimension*userNum)
		
	def updateParameters(self, articlePicked, click,  userID):	
		featureDimension = len(articlePicked.featureVector)
		T_X = vectorize(np.outer(articlePicked.featureVector, self.W.T[userID])) 
		self.T_A += np.outer(T_X, T_X)	
		self.T_b += click*T_X
		self.T_AInv = np.linalg.inv(self.T_A)

		Xi_Matirx = np.zeros(shape = (featureDimension, self.userNum))
		Xi_Matirx.T[userID] = articlePicked.featureVector
		W_X = vectorize( np.dot(np.transpose(self.UserTheta), Xi_Matirx))
		#print np.shape(W_X)
		self.W_A += np.outer(W_X, W_X)
		self.W_b += click * W_X
		self.W_AInv = np.linalg.inv(self.W_A)

		self.UserTheta = matrixize(np.dot(self.T_AInv, self.T_b), len(articlePicked.featureVector)) 
		self.W = matrixize(np.dot(self.W_AInv, self.W_b), self.userNum)
		#self.W = normalize(self.W, axis=0, norm='l1')

		#print 'A', self.W_A
		#print 'b', self.W_b
		'''
		plt.pcolor(self.W_b)
		plt.colorbar
		plt.show()
		'''

		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.BigW = np.kron(np.transpose(self.W), np.identity(n=len(articlePicked.featureVector)))
		self.CCA = np.dot(np.dot(self.BigW , self.T_AInv), np.transpose(self.BigW))

		self.BigTheta = np.kron(np.identity(n=self.userNum) , self.UserTheta)

		self.W_CCA = np.dot(np.dot(self.BigTheta , self.W_AInv), np.transpose(self.BigTheta))
	
	def getProb(self, alpha, article, userID):
		TempFeatureM = np.zeros(shape =(len(article.featureVector), self.userNum))
		TempFeatureM.T[userID] = article.featureVector
		TempFeatureV = vectorize(TempFeatureM)
		
		mean = np.dot(self.CoTheta.T[userID], article.featureVector)
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		
		W_var = np.sqrt(np.dot( np.dot(TempFeatureV, self.W_CCA) , TempFeatureV))

		pta = mean + alpha * (var + W_var)
		#pta = mean + alpha * var
		return pta



class W_W0_Struct(WStruct):
	def __init__(self, featureDimension, lambda_, eta_, userNum, W0):	
		WStruct.__init__(self, featureDimension, lambda_, eta_, userNum)
		self.W = W0

		self.CoTheta = np.dot(self.UserTheta, self.W)

		self.BigW = np.kron(np.transpose(self.W), np.identity(n=featureDimension))
	def updateParameters(self, articlePicked, click,  userID):	
		featureDimension = len(articlePicked.featureVector)
		T_X = vectorize(np.outer(articlePicked.featureVector, self.W.T[userID])) 
		self.T_A += np.outer(T_X, T_X)	
		self.T_b += click*T_X
		self.T_AInv = np.linalg.inv(self.T_A)

		Xi_Matirx = np.zeros(shape = (featureDimension, self.userNum))
		Xi_Matirx.T[userID] = articlePicked.featureVector
		W_X = vectorize( np.dot(np.transpose(self.UserTheta), Xi_Matirx))
		#print np.shape(W_X)
		self.W_A += np.outer(W_X, W_X)
		self.W_b += click * W_X
		self.W_AInv = np.linalg.inv(self.W_A)

		self.UserTheta = matrixize(np.dot(self.T_AInv, self.T_b), len(articlePicked.featureVector)) 
		#self.W = matrixize(np.dot(self.W_AInv, self.W_b), self.userNum)
		#self.W = normalize(self.W, axis=0, norm='l1')

		#print 'A', self.W_A
		#print 'b', self.W_b
		'''
		plt.pcolor(self.W_b)
		plt.colorbar
		plt.show()
		'''

		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.BigW = np.kron(np.transpose(self.W), np.identity(n=len(articlePicked.featureVector)))
		self.CCA = np.dot(np.dot(self.BigW , self.T_AInv), np.transpose(self.BigW))

		self.BigTheta = np.kron(np.identity(n=self.userNum) , self.UserTheta)

		self.W_CCA = np.dot(np.dot(self.BigTheta , self.W_AInv ), np.transpose(self.BigTheta))


class LearnW_W0_Struct(WStruct):
	def __init__(self, featureDimension, lambda_, eta_, userNum, W0):	
		WStruct.__init__(self, featureDimension, lambda_, eta_, userNum)
		self.W = W0

		self.CoTheta = np.dot(self.UserTheta, self.W)

		self.BigW = np.kron(np.transpose(self.W), np.identity(n=featureDimension))
	def updateParameters(self, articlePicked, click,  userID):	
		featureDimension = len(articlePicked.featureVector)
		T_X = vectorize(np.outer(articlePicked.featureVector, self.W.T[userID])) 
		self.T_A += np.outer(T_X, T_X)	
		self.T_b += click*T_X
		self.T_AInv = np.linalg.inv(self.T_A)

		Xi_Matirx = np.zeros(shape = (featureDimension, self.userNum))
		Xi_Matirx.T[userID] = articlePicked.featureVector
		W_X = vectorize( np.dot(np.transpose(self.UserTheta), Xi_Matirx))
		#print np.shape(W_X)
		self.W_A += np.outer(W_X, W_X)
		self.W_b += click * W_X
		self.W_AInv = np.linalg.inv(self.W_A)

		self.UserTheta = matrixize(np.dot(self.T_AInv, self.T_b), len(articlePicked.featureVector)) 
		self.W = matrixize(np.dot(self.W_AInv, self.W_b), self.userNum)
		#self.W = normalize(self.W, axis=0, norm='l1')

		#print 'A', self.W_A
		#print 'b', self.W_b
		'''
		plt.pcolor(self.W_b)
		plt.colorbar
		plt.show()
		'''

		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.BigW = np.kron(np.transpose(self.W), np.identity(n=len(articlePicked.featureVector)))
		self.CCA = np.dot(np.dot(self.BigW , self.T_AInv), np.transpose(self.BigW))

		self.BigTheta = np.kron(np.identity(n=self.userNum) , self.UserTheta)

		self.W_CCA = np.dot(np.dot(self.BigTheta , self.W_AInv), np.transpose(self.BigTheta))


	

class WAlgorithm:
	def __init__(self, dimension, alpha, lambda_, eta_, n):  # n is number of users
		self.USERS = WStruct(dimension, lambda_, eta_, n)
		self.dimension = dimension
		self.alpha = alpha

		self.CanEstimateUserPreference = True
		self.CanEstimateCoUserPreference =  True
		self.CanEstimateW = True

	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.USERS.getProb(self.alpha, x, userID)
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked
	def updateParameters(self, articlePicked, click, userID):
		self.USERS.updateParameters(articlePicked, click, userID)
		
	def getTheta(self, userID):
		return self.USERS.UserTheta.T[userID]

	def getCoTheta(self, userID):
		return self.USERS.CoTheta.T[userID]
	def getW(self, userID):
		#print self.USERS.W
		#return self.USERS.W.T[userID]
		return self.USERS.W.T

	def getA(self):
		return self.USERS.A
		
class W_W0_Algorithm(WAlgorithm):
	def __init__(self, dimension, alpha, lambda_, eta_, n, W0):  # n is number of users
		WAlgorithm.__init__(self, dimension, alpha, lambda_, eta_, n)
		self.USERS = W_W0_Struct(dimension, lambda_, eta_, n, W0)	
	def getW(self, userID):
		#print 'zzzzzzz',  self.USERS.W
		#return self.USERS.W.T[userID]
		return self.USERS.W.T


class LearnW_W0_Algorithm(WAlgorithm):
	def __init__(self, dimension, alpha, lambda_, eta_, n, W0):  # n is number of users
		WAlgorithm.__init__(self, dimension, alpha, lambda_, eta_, n)
		self.USERS = LearnW_W0_Struct(dimension, lambda_, eta_, n, W0)	
	def getW(self, userID):
		#print 'zzzzzzz',  self.USERS.W
		#return self.USERS.W.T[userID]
		return self.USERS.W.T

class WknowThetaAlgorithm(WAlgorithm):	
	def __init__(self, dimension, alpha, lambda_, eta_, n, theta):
		WAlgorithm.__init__(self, dimension, alpha, lambda_, eta_, n)
		self.USERS = WknowThetaStruct(dimension, eta_, n, theta)
		self.CanEstimateUserPreference = False
		self.CanEstimateCoUserPreference = False
		self.CanEstimateW = True
	def getW(self, userID):
		#print self.USERS.W
		#return self.USERS.W.T[userID]
		return self.USERS.W.T


