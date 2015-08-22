from conf import *  # it saves the address of data stored and where to save the data produced by algorithms
import time
import re           # regular expression library
from random import random, choice   # for random strategy
from operator import itemgetter
import datetime
import numpy as np  
from scipy.sparse import csgraph
from scipy.spatial import distance
#from YahooExp_util_functions import getClusters, getIDAssignment, parseLine, save_to_file, initializeW, vectorize, matrixize, articleAccess
from LastFM_util_functions import getFeatureVector, initializeW, initializeGW, parseLine, save_to_file


from CoLin import AsyCoLinUCBUserSharedStruct, AsyCoLinUCBAlgorithm, CoLinUCBUserSharedStruct
from LinUCB import LinUCBUserStruct
from GOBLin import GOBLinSharedStruct

# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
    def __init__(self):
        self.reward = 0

# structure to save data from LinUCB strategy
class LinUCBStruct(LinUCBUserStruct):
    def __init__(self, featureDimension, lambda_):
        LinUCBUserStruct.__init__(self, featureDimension= featureDimension, lambda_ = lambda_)
        self.reward = 0

# structure to save data from CoLinUCB strategy
class CoLinUCBStruct(AsyCoLinUCBUserSharedStruct):
    def __init__(self, featureDimension, lambda_, userNum, W):
        AsyCoLinUCBUserSharedStruct.__init__(self, featureDimension = featureDimension, lambda_ = lambda_, userNum = userNum, W = W)
        self.reward = 0  

class GOBLinStruct(GOBLinSharedStruct):
    def __init__(self, featureDimension, lambda_, userNum, W):
        GOBLinSharedStruct.__init__(self, featureDimension = featureDimension, lambda_ = lambda_, userNum = userNum, W = W)
        self.reward = 0
    

if __name__ == '__main__':
    # regularly print stuff to see if everything is going alright.
    # this function is inside main so that it shares variables with main and I dont wanna have large number of function arguments
    def printWrite():
        LinUCBTotalReward = 0
        for i in range(userNum): 
            LinUCBTotalReward += LinUCB_users[i].reward    

        print totalObservations
        print 'random', articles_random.reward,'  CoLin', CoLinUCB_USERS.reward, 'LinUCB', LinUCBTotalReward, 'GOBLin', GOBLin_USERS.reward   
        recordedStats = [articles_random.reward, CoLinUCB_USERS.reward, LinUCBTotalReward, GOBLin_USERS.reward]
        # write to file
        save_to_file(fileNameWrite, recordedStats, tim) 


    timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M')  # the current data time
    fileSig = 'LastFM_100'
    batchSize = 50                          # size of one batch
    
    d = 25           # feature dimension
    alpha = 0.3     # control how much to explore
    lambda_ = 0.2   # regularization used in matrix A
    Gepsilon = 0.3   # Parameter in initializing GW
    
    totalObservations = 0

 
    userNum = 100
    W = initializeW(userNum, relationFileName)   # Generate user relation matrix
    GW = initializeGW(Gepsilon,userNum,relationFileName)
    
    articles_random = randomStruct()
    CoLinUCB_USERS = CoLinUCBStruct(d, lambda_ ,userNum, W)
    GOBLin_USERS = GOBLinStruct(d, lambda_, userNum, GW)
    LinUCB_users = []  
    for i in range(userNum):
        LinUCB_users.append(LinUCBStruct(d, lambda_ ))
     
    fileName = LastFM_address + "/processed_events.dat"
    fileNameWrite = os.path.join(LastFM_save_address, fileSig + timeRun + '.csv')
    #FeatureVectorsFileName =  LastFM_address + '/Arm_FeatureVectors.dat'

    # put some new data in file for readability
    with open(fileNameWrite, 'a+') as f:
        f.write('\nNew Run at  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
        f.write('\n, Time,RandomReward; CoLinReward; LinUCBReward; GOBLinReward\n')

    print fileName, fileNameWrite

    with open(fileName, 'r') as f:
        LinUCBTotalReward  = 0
        # reading file line ie observations running one at a time
        for line in f:
            CoLinReward = 0
            LinUCBReward = 0
            GOBLinReward = 0

            totalObservations +=1
            userID, tim, pool_articles = parseLine(line)
            #tim, article_chosen, click, user_features, pool_articles = parseLine(line)
            #currentUser_featureVector = user_features[:-1]
            #currentUserID = getIDAssignment(np.asarray(currentUser_featureVector), userFeatureVectors)                
            
            currentArticles = []
            CoLinUCB_maxPTA = float('-inf')
            CoLinUCBPicked = None  
            LinUCB_maxPTA = float('-inf')  
            LinUCBPicked = None  
            GOBLin_maxPTA = float('-inf')
            GOBLinPicked = None  
           
            currentUserID = userID
            if currentUserID > userNum:
                continue
            else:  
                article_chosen = int(pool_articles[0])  
                #for article in np.random.permutation(pool_articles) :
                for article in pool_articles:
                    article_id = int(article.strip(']'))
                    #print article_id
                    article_featureVector = getFeatureVector(FeatureVectorsFileName, article_id)
                    article_featureVector =np.array(article_featureVector ,dtype=float)
                    #print article_featureVector
                    currentArticles.append(article_id)
                    # CoLinUCB pick article
                    if len(article_featureVector)==25:
                        #print 'Yes'
                        CoLinUCB_pta = CoLinUCB_USERS.getProb(alpha, article_featureVector, currentUserID)
                        #print article_id, CoLinUCB_pta
                        if CoLinUCB_maxPTA < CoLinUCB_pta:
                            CoLinUCBPicked = article_id    # article picked by CoLinUCB
                            CoLinUCB_PickedfeatureVector = article_featureVector
                            CoLinUCB_maxPTA = CoLinUCB_pta
                            #print CoLinUCBPicked
                        LinUCB_pta = LinUCB_users[currentUserID].getProb(alpha, article_featureVector)
                        if LinUCB_maxPTA < LinUCB_pta:
                            LinUCBPicked = article_id
                            LinUCB_PickedfeatureVector =  article_featureVector
                            LinUCB_maxPTA = LinUCB_pta
                        GOBLin_pta = GOBLin_USERS.getProb(alpha, article_featureVector, currentUserID)
                        if GOBLin_maxPTA < GOBLin_pta:
                            GOBLinPicked = article_id    # article picked by GOB.Lin
                            GOBLin_PickedfeatureVector = article_featureVector
                            GOBLin_maxPTA = GOBLin_pta

                # article picked by random strategy
                #article_chosen = currentArticles[0]
                #print article_chosen, CoLinUCBPicked, LinUCBPicked, GOBLinPicked
                if CoLinUCBPicked !=LinUCBPicked:
                    print 'Error!!!!!'
                RandomPicked = choice(currentArticles)
                if RandomPicked == article_chosen:
                    articles_random.reward +=1

                if CoLinUCBPicked == article_chosen:
                    CoLinUCB_USERS.reward +=1
                    CoLinReward = 1
                CoLinUCB_USERS.updateParameters(CoLinUCB_PickedfeatureVector,CoLinReward, currentUserID)

                if LinUCBPicked == article_chosen:
                    LinUCB_users[currentUserID].reward +=1
                    LinUCBReward = 1
                LinUCB_users[currentUserID].updateParameters(LinUCB_PickedfeatureVector, LinUCBReward)

                if GOBLinPicked == article_chosen:
                    GOBLin_USERS.reward +=1
                    GOBLinReward = 1
                GOBLin_USERS.updateParameters(GOBLin_PickedfeatureVector, GOBLinReward, currentUserID)
                # if the batch has ended
                if totalObservations%batchSize==0:
                    printWrite()
        #print stuff to screen and save parameters to file when the Yahoo! dataset file ends
        printWrite()
