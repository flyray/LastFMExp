from conf import * 	# it saves the address of data stored and where to save the data produced by algorithms
import argparse # For argument parsing
import time
import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter
import datetime
import numpy as np 	
import sys
from scipy.sparse import csgraph
from scipy.spatial import distance
from YahooExp_util_functions import *


from CoLin import *
from GOBLin import *
from LinUCB import *


# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
	def __init__(self):
		self.learn_stats = articleAccess()

# structure to save data from CoLinUCB strategy
class CoLinUCBStruct(AsyCoLinUCBUserSharedStruct):
	def __init__(self, featureDimension, lambda_, articleNum, W):
		AsyCoLinUCBUserSharedStruct.__init__(self, featureDimension = featureDimension, lambda_ = lambda_, userNum = articleNum, W = W)
		self.learn_stats = articleAccess()	

class GOBLinStruct(GOBLinSharedStruct):
	def __init__(self, featureDimension, lambda_, articleNum, W):
		GOBLinSharedStruct.__init__(self, featureDimension = featureDimension, lambda_ = lambda_, userNum = articleNum, W = W)
		self.learn_stats = articleAccess()	
class LinUCBStruct(LinUCBUserStruct):
	def __init__(self, featureDimension, lambda_):
		LinUCBUserStruct.__init__(self, featureDimension= featureDimension, lambda_ = lambda_)
		self.learn_stats = articleAccess()

if __name__ == '__main__':
	# regularly print stuff to see if everything is going alright.
	# this function is inside main so that it shares variables with main and I dont wanna have large number of function arguments
	def printWrite():
		randomLearnCTR = articles_random.learn_stats.updateCTR()
		if algName == 'CoLin':
			CoLinUCBCTR = CoLinUCB_ARTICLES.learn_stats.updateCTR()
			print totalObservations
			print 'random', randomLearnCTR,'  CoLin', CoLinUCBCTR
			recordedStats = [randomLearnCTR, CoLinUCBCTR, CoLinUCB_ARTICLES.learn_stats.accesses, CoLinUCB_ARTICLES.learn_stats.clicks]	
		if algName =='GOBLin':
			GOBLinCTR = GOBLin_ARTICLES.learn_stats.updateCTR()
			print totalObservations
			print 'random', randomLearnCTR,'  GOBLin', GOBLinCTR  	
			recordedStats = [randomLearnCTR, GOBLinCTR, GOBLin_ARTICLES.learn_stats.accesses, GOBLin_ARTICLES.learn_stats.clicks]
		if algName == 'Uniform_LinUCB':
			UniformLinUCBCTR = LinUCB_ARTICLES.learn_stats.updateCTR()
			print totalObservations
			print 'random', randomLearnCTR, 'Uniform_LinUCB', UniformLinUCBCTR
			recordedStats =[randomLearnCTR, UniformLinUCBCTR, LinUCB_ARTICLES.learn_stats.accesses,LinUCB_ARTICLES.learn_stats.clicks]
		if algName == 'LinUCB':
			TotalLinUCBAccess = 0.0
			TotalLinUCBClick = 0.0
			for i in range(articleNum):			
				TotalLinUCBAccess += LinUCB_Articles [i].learn_stats.accesses
				TotalLinUCBClick += LinUCB_Articles[i].learn_stats.clicks

			if TotalLinUCBAccess != 0:
				LinUCBCTR = TotalLinUCBClick/(1.0*TotalLinUCBAccess)
			else:
				LinUCBCTR = -1.0

			print totalObservations
			print 'random', randomLearnCTR,'	LinUCB', LinUCBCTR
		
			recordedStats = [randomLearnCTR,  LinUCBCTR, TotalLinUCBAccess, TotalLinUCBClick]
			# write to file
		save_to_file(fileNameWrite, recordedStats, tim) 
	def WriteStat():
		with open(fileNameWriteStatTP, 'a+') as f:
			for key, val in articleTruePositve.items():
				f.write(str(key) + ';'+str(val) + ',')
			f.write('\n')
		with open(fileNameWriteStatTN, 'a+') as f:
			for key, val in articleTrueNegative.items():
				f.write(str(key) + ';'+str(val) + ',')
			f.write('\n')
		with open(fileNameWriteStatFP, 'a+') as f:
			for key, val in articleFalsePositive.items():
				f.write(str(key) + ';'+str(val) + ',')
			f.write('\n')


	def calculateStat():
		if click:		
			for article in currentArticles:
				if article == article_chosen:
					articleTruePositve[article_chosen] +=1
				else:
					articleTrueNegative[article] +=1				
		else:
			for article in currentArticles:
				if article == article_chosen:
					articleFalsePositive[article_chosen] +=1

	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--YahooDataFile', dest="Yahoo_save_address", help="input the adress for Yahoo data")
	parser.add_argument('--ArticleDictFile', dest="ArticleDictFilename", help="input the file for Article FeatureVectors")
	parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLin, GOBLin, LinUCB, or Uniform_LinUCB')

	parser.add_argument('--showheatmap', action='store_true',
	                help='Show heatmap of relation matrix.') 
	#parser.add_argument('--userNum', dest = 'userNum', help = 'Set the userNum, can be 20, 40, 80, 160')

	parser.add_argument('--Sparsity', dest = 'SparsityLevel', help ='Set the SparsityLevel by choosing the top M most connected ARTICLES, should be smaller than userNum, when equal to userNum, we are using a full connected graph')
	parser.add_argument('--diag', dest="DiagType", help="Specify the setting of diagional setting, can be set as 'Orgin' or 'Opt' ") 


	args = parser.parse_args()

	algName = str(args.alg)
	#clusterNum = int(args.userNum)
	SparsityLevel = int(args.SparsityLevel)
	yahooData_address = str(args.Yahoo_save_address)
	DiagType = str(args.DiagType)
	ArticleDictFilename = str(args.ArticleDictFilename)
	        
	timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 	# the current data time
	dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
	fileSig = 'ArticleSide' + str(DiagType)+ 'SP'+ str(SparsityLevel)+algName
	batchSize = 2000
	statBatchSize = 200000							# size of one batch
	
	d = 5 	        # feature dimension
	alpha = 0.3     # control how much to explore
	lambda_ = 0.2   # regularization used in matrix A
	epsilon = 0.3
	totalObservations = 0

	articleTruePositve = {}
	articleFalseNegative = {}

	articleTrueNegative = {}
	articleFalsePositive = {}

	articleDic =  getArticleDic(ArticleDictFilename)  # get article information from dictionary
	articleIDlist = articleDic.keys()
	articleFeatureVector = articleDic.values()
	articleFeatureVector = np.asarray(articleFeatureVector)
	articleNum = len(articleFeatureVector) # not exactly
	if DiagType == 'Orgin':
		W = initializeW(articleFeatureVector, SparsityLevel)
	elif DiagType == 'Opt':
		W = initializeW_opt(articleFeatureVector, SparsityLevel)   # Generate user relation matrix
	GW = initializeGW(W , epsilon)
	print W
 	
	articles_random = randomStruct()
	CoLinUCB_ARTICLES = CoLinUCBStruct(d, lambda_ ,articleNum, W )
	GOBLin_ARTICLES = GOBLinStruct(d, lambda_, articleNum, GW)
	LinUCB_ARTICLES = LinUCBStruct(d, lambda_)
	LinUCB_Articles = {}
	for i in range(articleNum):
		LinUCB_Articles[i] = LinUCBStruct(d, lambda_ )
	
	for dataDay in dataDays:
		fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay
		fileNameWrite = os.path.join(Yahoo_save_address, fileSig + dataDay + timeRun + '.csv')

		fileNameWriteStatTP = os.path.join(Yahoo_save_address, 'Stat_TP'+ fileSig + dataDay + timeRun + '.csv')
		fileNameWriteStatTN = os.path.join(Yahoo_save_address, 'Stat_TN'+ fileSig + dataDay + timeRun + '.csv')
		fileNameWriteStatFP = os.path.join(Yahoo_save_address, 'Stat_FP'+ fileSig + dataDay + timeRun + '.csv')

		# put some new data in file for readability
		with open(fileNameWrite, 'a+') as f:
			f.write('\nNewRunat  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
			f.write('\n,Time,RandomCTR;'+ str(algName) + 'CTR;' + 'accesses;'+ 'clicks;' + '' +'\n')

		print fileName, fileNameWrite
		with open(fileName, 'r') as f:
			# reading file line ie observations running one at a time
			for line in f:
				totalObservations +=1

				tim, article_chosen, click, user_features, pool_articles = parseLine(line)
				currentUser_featureVector = user_features[:-1]
				currentArticles = []

				CoLinUCB_maxPTA = float('-inf')
				CoLinUCBPicked = None

				GOBLin_maxPTA = float('-inf')
				GOBLinPicked = None

				UniformLinUCB_maxPTA = float('-inf')
				UniformLinUCBPicked = None

				LinUCB_maxPTA = float('-inf')
				LinUCBPicked = None
				for article in pool_articles:
					article_id = int(article[0])
					if article_id in articleDic:
						article_idIndex = articleIDlist.index(article_id)
						article_featureVector =np.asarray(article[1:6])
						currentArticles.append(article_id)
						if algName == 'CoLin':
							CoLinUCB_pta = CoLinUCB_ARTICLES.getProb(alpha, currentUser_featureVector, article_idIndex)
							if CoLinUCB_maxPTA < CoLinUCB_pta:
								CoLinUCBPicked = article_idIndex    # article picked by CoLinUCB
								CoLinUCB_maxPTA = CoLinUCB_pta
						if algName == 'GOBLin':
							GOBLin_pta = GOBLin_ARTICLES.getProb(alpha, currentUser_featureVector, article_idIndex)
							if GOBLin_maxPTA < GOBLin_pta:
								GOBLinPicked = article_idIndex    # article picked by GOB.Lin
								GOBLin_maxPTA = GOBLin_pta
						if algName == 'Uniform_LinUCB':
							UniformLinUCB_pta = LinUCB_ARTICLES.getProb(alpha, currentUser_featureVector)
							if UniformLinUCB_maxPTA < UniformLinUCB_pta:
								UniformLinUCBPicked = article_idIndex
								UniformLinUCB_maxPTA = UniformLinUCB_pta
						if algName == 'LinUCB':
							LinUCB_pta = LinUCB_Articles[article_idIndex].getProb(alpha, currentUser_featureVector)
							if LinUCB_maxPTA < LinUCB_pta:
								LinUCBPicked = article_idIndex    # article picked by CoLinU
								LinUCB_maxPTA = LinUCB_pta
				for article in currentArticles:
					if article not in articleTruePositve:
						articleTruePositve[article] = 0
						articleTrueNegative[article] = 0
						articleFalsePositive[article] = 0
						articleFalseNegative[article] = 0
			#print  GOBLinPicked
			#time.sleep(1)
				# article picked by random strategy
				articles_random.learn_stats.addrecord(click)
				if article_chosen in articleDic:
					article_chosen = articleIDlist.index(article_chosen)   # get the index of article_choseen
					if algName == 'CoLin':
						if CoLinUCBPicked == article_chosen:
							CoLinUCB_ARTICLES.learn_stats.addrecord(click)
							CoLinUCB_ARTICLES.updateParameters(currentUser_featureVector, click, CoLinUCBPicked)
							calculateStat()
					if algName == 'GOBLin':
						if GOBLinPicked == article_chosen:
							GOBLin_ARTICLES.learn_stats.addrecord(click)
							GOBLin_ARTICLES.updateParameters(currentUser_featureVector, click, GOBLinPicked)
							calculateStat()
					if algName == 'Uniform_LinUCB':
						if UniformLinUCBPicked == article_chosen:
							LinUCB_ARTICLES.learn_stats.addrecord(click)
							LinUCB_ARTICLES.updateParameters(currentUser_featureVector, click)
							calculateStat()
					if algName == 'LinUCB':
						if LinUCBPicked == article_chosen:
							LinUCB_Articles[LinUCBPicked].learn_stats.addrecord(click)
							LinUCB_Articles[LinUCBPicked].updateParameters(currentUser_featureVector, click)
							calculateStat()
					# if the batch has ended
					if totalObservations%batchSize==0:
						printWrite()
				'''
				if totalObservations%statBatchSize==0:
					WriteStat()
				'''
			#print stuff to screen and save parameters to file when the Yahoo! dataset file ends
			printWrite()
			WriteStat()
