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


from CoLin import AsyCoLinUCBUserSharedStruct, AsyCoLinUCBAlgorithm, CoLinUCBUserSharedStruct
from GOBLin import GOBLinSharedStruct
from LinUCB import LinUCBUserStruct


# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
	def __init__(self):
		self.learn_stats = articleAccess()

# structure to save data from CoLinUCB strategy
class CoLinUCBStruct(AsyCoLinUCBUserSharedStruct):
	def __init__(self, featureDimension, lambda_, userNum, W):
		AsyCoLinUCBUserSharedStruct.__init__(self, featureDimension = featureDimension, lambda_ = lambda_, userNum = userNum, W = W)
		self.learn_stats = articleAccess()	

class GOBLinStruct(GOBLinSharedStruct):
	def __init__(self, featureDimension, lambda_, userNum, W):
		GOBLinSharedStruct.__init__(self, featureDimension = featureDimension, lambda_ = lambda_, userNum = userNum, W = W)
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
			CoLinUCBCTR = CoLinUCB_USERS.learn_stats.updateCTR()
			print totalObservations
			print 'random', randomLearnCTR,'  CoLin', CoLinUCBCTR
			recordedStats = [randomLearnCTR, CoLinUCBCTR, CoLinUCB_USERS.learn_stats.accesses, CoLinUCB_USERS.learn_stats.clicks]	
		if algName =='GOBLin':
			GOBLinCTR = GOBLin_USERS.learn_stats.updateCTR()
			print totalObservations
			print 'random', randomLearnCTR,'  GOBLin', GOBLinCTR  	
			recordedStats = [randomLearnCTR, GOBLinCTR, GOBLin_USERS.learn_stats.accesses, GOBLin_USERS.learn_stats.clicks]
                if algName == 'Uniform_LinUCB':
                        UniformLinUCBCTR = LinUCB_USERS.learn_stats.updateCTR()
                        print totalObservations
                        print 'random', randomLearnCTR, 'Uniform_LinUCB', UniformLinUCBCTR
                        recordedStats =[randomLearnCTR, UniformLinUCBCTR, LinUCB_USERS.learn_stats.accesses,LinUCB_USERS.learn_stats.clicks]
                if algName == 'LinUCB':
			TotalLinUCBAccess = 0.0
			TotalLinUCBClick = 0.0
			for i in range(userNum):			
				TotalLinUCBAccess += LinUCB_users[i].learn_stats.accesses
				TotalLinUCBClick += LinUCB_users[i].learn_stats.clicks
	
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
        parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLin, GOBLin, LinUCB, or Uniform_LinUCB')
   
        parser.add_argument('--showheatmap', action='store_true',
                        help='Show heatmap of relation matrix.') 
        parser.add_argument('--userNum', dest = 'userNum', help = 'Set the userNum, can be 20, 40, 80, 160')

        parser.add_argument('--Sparsity', dest = 'SparsityLevel', help ='Set the SparsityLevel by choosing the top M most connected users, should be smaller than userNum, when equal to userNum, we are using a full connected graph')
        parser.add_argument('--diag', dest="DiagType", help="Specify the setting of diagional setting, can be set as 'Orgin' or 'Opt' ") 


        args = parser.parse_args()
    
        algName = str(args.alg)
        clusterNum = int(args.userNum)
        SparsityLevel = int(args.SparsityLevel)
        yahooData_address = str(args.Yahoo_save_address)
        DiagType = str(args.DiagType)


	timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 	# the current data time
	dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
	fileSig = str(DiagType)+str(clusterNum)+ 'SP'+ str(SparsityLevel)+algName
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

	fileNameWriteCluster = os.path.join(Kmeansdata_address, '10kmeans_model'+str(clusterNum)+ '.dat')
	userFeatureVectors = getClusters(fileNameWriteCluster)	
	userNum = clusterNum
	if DiagType == 'Orgin':
		W = initializeW(userFeatureVectors, SparsityLevel)
	elif DiagType == 'Opt':
		W = initializeW_opt(userFeatureVectors, SparsityLevel)   # Generate user relation matrix
	GW = initializeGW(W , epsilon)
 	
	articles_random = randomStruct()
	CoLinUCB_USERS = CoLinUCBStruct(d, lambda_ ,userNum, W )
	GOBLin_USERS = GOBLinStruct(d, lambda_, userNum, GW)
        LinUCB_USERS = LinUCBStruct(d, lambda_)
        LinUCB_users = []
	for i in range(userNum):
		LinUCB_users.append(LinUCBStruct(d, lambda_ ))
	
	for dataDay in dataDays:
		fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay	+'.userID'
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

				tim, article_chosen, click, currentUserID, pool_articles = parseLine_ID(line)
				#currentUser_featureVector = user_features[:-1]
				#currentUserID = getIDAssignment(np.asarray(currentUser_featureVector), userFeatureVectors)                
                
                                #-----------------------------Pick an article (CoLinUCB, LinUCB, Random)-------------------------
                                currentArticles = []
                                CoLinUCB_maxPTA = float('-inf')
                                CoLinUCBPicked = None      
                                CoLinUCB_PickedfeatureVector = np.array([0,0,0,0,0])

                                GOBLin_maxPTA = float('-inf')
                                GOBLinPicked = None
                                GOBLin_PickedfeatureVector = np.array([0,0,0,0,0])

                                UniformLinUCB_maxPTA = float('-inf')
                                UniformLinUCBPicked = None
                                UniformLiUCB_PickedfeatureVector = np.array([0,0,0,0,0])

                             	LinUCB_maxPTA = float('-inf')  
                                LinUCBPicked = None
                                LinUCB_PickedfeatureVector = np.array([0,0,0,0,0])

                                for article in pool_articles:
                                        article_id = int(article[0])
                                        article_featureVector =np.asarray(article[1:6])
                                        currentArticles.append(article_id)
	                                        # CoLinUCB pick article
                                        if len(article_featureVector)==5:
                                                if algName == 'CoLin':
                                                        CoLinUCB_pta = CoLinUCB_USERS.getProb(alpha, article_featureVector, currentUserID)
                                                        if CoLinUCB_maxPTA < CoLinUCB_pta:
                                                                CoLinUCBPicked = article_id    # article picked by CoLinUCB
                                                                CoLinUCB_PickedfeatureVector = article_featureVector
                                                                CoLinUCB_maxPTA = CoLinUCB_pta
                                                if algName == 'GOBLin':
                                                        GOBLin_pta = GOBLin_USERS.getProb(alpha, article_featureVector, currentUserID)
                                                        if GOBLin_maxPTA < GOBLin_pta:
                                                                GOBLinPicked = article_id    # article picked by GOB.Lin
                                                                GOBLin_PickedfeatureVector = article_featureVector
                                                                GOBLin_maxPTA = GOBLin_pta
                                                if algName == 'Uniform_LinUCB':
                                                        UniformLinUCB_pta = LinUCB_USERS.getProb(alpha, article_featureVector)
                                                        if UniformLinUCB_maxPTA < UniformLinUCB_pta:
                                                                UniformLinUCBPicked = article_id
                                                                UniformLinUCB_PickedfeatureVector = article_featureVector
                                                                UniformLinUCB_maxPTA = UniformLinUCB_pta
                                                                 
                                                if algName == 'LinUCB':
                                                        LinUCB_pta = LinUCB_users[currentUserID].getProb(alpha, article_featureVector)
	                                                if LinUCB_maxPTA < LinUCB_pta:
                                                                LinUCBPicked = article_id    # article picked by CoLinU
	                                                        LinUCB_PickedfeatureVector = article_featureVector
	                                                        LinUCB_maxPTA = LinUCB_pta
					print article_id, LinUCB_pta
				time.sleep(1)
                                for article in currentArticles:
                                	if article not in articleTruePositve:
                                            articleTruePositve[article] = 0
                                            articleTrueNegative[article] = 0
                                            articleFalsePositive[article] = 0
                                            articleFalseNegative[article] = 0

                               
                                # article picked by random strategy
                                articles_random.learn_stats.addrecord(click)
                                if algName == 'CoLin':
	                                if CoLinUCBPicked == article_chosen:
                                                CoLinUCB_USERS.learn_stats.addrecord(click)
                                                CoLinUCB_USERS.updateParameters(CoLinUCB_PickedfeatureVector, click, currentUserID)
                                               	calculateStat()
                                                
                                if algName == 'GOBLin':
	                            	if GOBLinPicked == article_chosen:
                                                GOBLin_USERS.learn_stats.addrecord(click)
                                                GOBLin_USERS.updateParameters(GOBLin_PickedfeatureVector, click, currentUserID)
                                                calculateStat()
                                if algName == 'Uniform_LinUCB':
                                        if UniformLinUCBPicked == article_chosen:
                                                LinUCB_USERS.learn_stats.addrecord(click)
                                                LinUCB_USERS.updateParameters(UniformLinUCB_PickedfeatureVector, click)
                                                calculateStat()
                                if algName == 'LinUCB':
					print 'Picked', LinUCBPicked, click,LinUCB_maxPTA, article_chosen
                                	if LinUCBPicked == article_chosen:
						
                                                LinUCB_users[currentUserID].learn_stats.addrecord(click)
                                                LinUCB_users[currentUserID].updateParameters(LinUCB_PickedfeatureVector, click)
                                                calculateStat()
                                # if the batch has ended
                                if totalObservations%batchSize==0:
                                        printWrite()
                                if totalObservations%statBatchSize==0:
                                        WriteStat()
                        #print stuff to screen and save parameters to file when the Yahoo! dataset file ends
                        printWrite()
                        WriteStat()
