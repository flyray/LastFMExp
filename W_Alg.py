import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.optimize import minimize
import math
from util_functions import vectorize, matrixize
import time
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from numpy import linalg as LA
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp

def vectorize(M):
	temp = []
	for i in range(M.shape[0]*M.shape[1]):
		temp.append(M.T.item(i))
	V = np.asarray(temp)
	return V

def matrixize(V, C_dimension):
	temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	for i in range(len(V)/C_dimension):
		temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	W = temp
	return W

def getcons(dim):
	cons = []
	cons.append({'type': 'eq','fun': lambda x : np.sum(x)-1})
	for i in range(dim):
		cons.append({'type' : 'ineq','fun' : lambda  x: x[i] })
	return tuple(cons)
def getbounds(dim):
	bnds = []
	for i in range(dim):
		bnds.append((0,1))
	return tuple(bnds)


class WStruct_batch_Cons:
	def __init__(self, featureDimension, lambda_,userNum,W, windowSize, RankoneInverse, WRegu):
		self.windowSize = windowSize
		self.counter = 1
		self.RankoneInverse = RankoneInverse
		self.WRegu = WRegu
		self.userNum = userNum
		self.lambda_ = lambda_
		# Basic stat in estimating Theta
		self.A = lambda_*np.identity(n = featureDimension*userNum)
		self.b = np.zeros(featureDimension*userNum)
		self.UserTheta = np.zeros(shape = (featureDimension, userNum))
		#self.UserTheta = np.random.random((featureDimension, userNum))
		self.AInv = np.linalg.inv(self.A)
		
		#self.W = np.random.random((userNum, userNum))
		#self.W = np.identity(n = userNum)
		self.W = W
		self.Wlong = vectorize(self.W)
		self.batchGradient = np.zeros(userNum*userNum)

		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.BigW = np.kron(np.transpose(self.W), np.identity(n=featureDimension))
		self.CCA = np.identity(n = featureDimension*userNum)
		self.BigTheta = np.kron(np.identity(n=userNum) , self.UserTheta)
		self.W_X_arr = []
		self.W_y_arr = []
		for i in range(userNum):
			self.W_X_arr.append([])
			self.W_y_arr.append([])
		
	def updateParameters(self, featureVector, click,  userID):	
		self.counter +=1
		self.Wlong = vectorize(self.W)
		featureDimension = len(featureVector)
		T_X = vectorize(np.outer(featureVector, self.W.T[userID])) 
		self.A += np.outer(T_X, T_X)	
		self.b += click*T_X
		if self.RankoneInverse:
			temp = np.dot(self.AInv, T_X)
			self.AInv = self.AInv - (np.outer(temp,temp))/(1.0+np.dot(np.transpose(T_X),temp))
		else:
			self.AInv =  np.linalg.inv(self.A)
		self.UserTheta = matrixize(np.dot(self.AInv, self.b), len(featureVector)) 

		Xi_Matirx = np.zeros(shape = (featureDimension, self.userNum))
		Xi_Matirx.T[userID] = featureVector
		W_X = vectorize( np.dot(np.transpose(self.UserTheta), Xi_Matirx))
		W_X_current = np.dot(np.transpose(self.UserTheta), featureVector)

		self.W_X_arr[userID].append(W_X_current)
		self.W_y_arr[userID].append(click)

		#print self.windowSize
		if self.counter%self.windowSize ==0:
			for i in range(len(self.W)):
				if len(self.W_X_arr[i]) !=0:
					def fun(w):
						w = np.asarray(w)
						return np.sum((np.dot(self.W_X_arr[i], w) - self.W_y_arr[i])**2, axis = 0) + self.lambda_*np.linalg.norm(w)**2
					def evaluateGradient(w):
						w = np.asarray(w)
						X = np.asarray(self.W_X_arr[i])
						y = np.asarray(self.W_y_arr[i])
						grad = np.dot(np.transpose(X) , ( np.dot(X,w)- y)) + self.lambda_ * w
						return 2*grad
					def fun_WRegu(w):
						w = np.asarray(w)
						return np.sum((np.dot(self.W_X_arr[i], w) - self.W_y_arr[i])**2, axis = 0) + self.lambda_*np.linalg.norm(w - self.W.T[i])**2
					def evaluateGradient_WRegu(w):
						w = np.asarray(w)
						X = np.asarray(self.W_X_arr[i])
						y = np.asarray(self.W_y_arr[i])
						grad = np.dot(np.transpose(X) , ( np.dot(X,w)- y)) + self.lambda_ * (w - self.W.T[i])
						return 2*grad
					current = self.W.T[i]
					if self.WRegu:
						res = minimize(fun_WRegu, current, constraints = getcons(len(self.W)), method ='SLSQP', jac = evaluateGradient_WRegu, bounds=getbounds(len(self.W)), options={'disp': False})
					else:
						res = minimize(fun, current, constraints = getcons(len(self.W)), method ='SLSQP', jac = evaluateGradient, bounds=getbounds(len(self.W)), options={'disp': False})
					self.W.T[i] = res.x
					if self.windowSize<2000:
						self.windowSize = self.windowSize*2 
		self.CoTheta = np.dot(self.UserTheta, self.W)
		self.BigW = np.kron(np.transpose(self.W), np.identity(n=len(featureVector)))
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))

		self.BigTheta = np.kron(np.identity(n=self.userNum) , self.UserTheta)
	def getProb(self, alpha, featureVector, userID):
		TempFeatureM = np.zeros(shape =(len(featureVector), self.userNum))
		TempFeatureM.T[userID] = featureVector
		TempFeatureV = vectorize(TempFeatureM)
		
		mean = np.dot(self.CoTheta.T[userID], featureVector)
		var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
		pta = mean + alpha * var
		#pta = mean + alpha * var
		return pta


	
class LearnWAlgorithm:
	def __init__(self, dimension, alpha, lambda_, eta_, n, windowSize, RankoneInverse = False, WRegu = False):  # n is number of users
		self.USERS = WStruct_batch_Cons(dimension, lambda_, eta_, n, windowSize, RankoneInverse, WRegu)
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
	def updateParameters(self, featureVector, click, userID):
		self.USERS.updateParameters(featureVector, click, userID)
		
	def getTheta(self, userID):
		return self.USERS.UserTheta.T[userID]

	def getCoTheta(self, userID):
		return self.USERS.CoTheta.T[userID]
	def getW(self, userID):
		#print self.USERS.W
		return self.USERS.W.T[userID]

	def getA(self):
		return self.USERS.A



