import numpy as np
from LinUCB import *
import math
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
class CLUBUserStruct(LinUCBUserStruct):
	def __init__(self,featureDimension, lambda_):
		LinUCBUserStruct.__init__(self,featureDimension = featureDimension,lambda_= lambda_)
		self.reward = 0
		self.CA = self.A
		self.Cb = self.b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv, self.Cb)
		self.I = lambda_*np.identity(n = featureDimension)	
		self.counter = 0
		self.CBPrime = 0
	def updateParameters(self, articlePicked_FeatureVector, click,alpha_2):
		#LinUCBUserStruct.updateParameters(self, articlePicked_FeatureVector, click)
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.dot(self.AInv, self.b)
		self.counter+=1
		self.CBPrime = alpha_2*np.sqrt((1+math.log10(1+self.counter))/(1+self.counter))

	def updateParametersofClusters(self,clusters,userID,Graph,users):
		self.CA = self.I
		self.Cb = np.zeros(self.d)
		#print type(clusters)

		for i in range(len(clusters)):
			if clusters[i] == clusters[userID]:
				self.CA += float(Graph[userID,i])*(users[i].A - self.I)
				self.Cb += float(Graph[userID,i])*users[i].b
				self.CAInv = np.linalg.inv(self.CA)
				self.CTheta = np.dot(self.CAInv,self.Cb)

	def getProb(self, alpha, article_FeatureVector,time):
		mean = np.dot(self.CTheta, article_FeatureVector)
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.CAInv),  article_FeatureVector))
		pta = mean + alpha * var*np.sqrt(math.log10(time+1))
		return pta

class CLUBAlgorithm(N_LinUCBAlgorithm):
	def __init__(self,dimension,alpha,lambda_,n):
		self.time = 0
		#N_LinUCBAlgorithm.__init__(dimension = dimension, alpha=alpha,lambda_ = lambda_,n=n)
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(n):
			self.users.append(CLUBUserStruct(dimension,lambda_)) 

		self.dimension = dimension
		self.alpha = alpha
		self.Graph = np.ones([n,n]) 
		self.clusters = []
		g = csr_matrix(self.Graph)
		N_components, components = connected_components(g)
			
	def decide(self,pool_articles,userID):
		self.users[userID].updateParametersofClusters(self.clusters,userID,self.Graph, self.users)
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, x.featureVector,self.time)
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x.id
				featureVectorPicked = x.featureVector
				maxPTA = x_pta
		self.time +=1

		return featureVectorPicked, articlePicked
	def updateParameters(self, featureVector, click,userID):
		self.users[userID].updateParameters(featureVector, click, self.alpha)
	def updateGraphClusters(self,userID):
		n = len(self.users)
		for j in range(n):
			ratio = float(np.linalg.norm(self.users[userID].UserTheta - self.users[j].UserTheta,1))/float(self.users[userID].CBPrime + self.users[j].CBPrime)
			if ratio > 1:
				ratio = 0
			self.Graph[userID][j] = ratio
			self.Graph[j][userID] = self.Graph[userID][j]
		N_components, component_list = connected_components(csr_matrix(self.Graph))
		self.clusters = component_list





