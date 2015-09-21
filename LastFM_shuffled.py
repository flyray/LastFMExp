import matplotlib.pyplot as plt
import argparse # For argument parsing
import os.path
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
from LastFM_util_functions_2 import *#getFeatureVector, initializeW, initializeGW, parseLine, save_to_file, initializeW_clustering, initializeGW_clustering
#from LastFM_util_functions import getFeatureVector, initializeW, initializeGW, parseLine, save_to_file

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
    
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

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
    
    # Introduce argparse for future argument parsing.
    parser = argparse.ArgumentParser(description='')

    # If exsiting cluster label data.
    parser.add_argument('--clusterfile', dest="clusterfile", help="input an clustering label file", 
                        metavar="FILE", type=lambda x: is_valid_file(parser, x))
    # Select algorithm.
    parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLinUCB, GOBLin, LinUCB or ALL.')
   
    # Designate relation matrix diagnol.
    parser.add_argument('--diagnol', dest='diagnol', required=True,
                        help='Designate relation matrix diagnol, could be any real number(in general this number should be small, for example around 0.2 when userNum is 50, around 0.1 when userNum is 100, around 0.05 when userNum is 200 ), or Origin.') 
    # Whether show heatmap of relation matrix.
    parser.add_argument('--showheatmap', action='store_true',
                        help='Show heatmap of relation matrix.') 
    # Dataset.
    parser.add_argument('--dataset', required=True, choices=['LastFM', 'Delicious'],
                        help='Select Dataset to run, could be LastFM or Delicious.')

    args = parser.parse_args()
    
    batchSize = 50                          # size of one batch
    
    d = 25           # feature dimension
    alpha = 0.3     # control how much to explore
    lambda_ = 0.2   # regularization used in matrix A
    Gepsilon = 0.3   # Parameter in initializing GW
    
    totalObservations = 0

 
    OriginaluserNum = 2100
    nClusters = 100
    userNum = nClusters   
    if args.dataset == 'LastFM':
        relationFileName = LastFM_relationFileName
        address = LastFM_address
        save_address = LastFM_save_address
        FeatureVectorsFileName = LastFM_FeatureVectorsFileName
    else:
        relationFileName = Delicious_relationFileName
        address = Delicious_address
        save_address = Delicious_save_address
        FeatureVectorsFileName = Delicious_FeatureVectorsFileName        
    if args.clusterfile:           
        label = read_cluster_label(args.clusterfile)
        userNum = nClusters = int(args.clusterfile.name.split('.')[-1]) # Get cluster number.
        W = initializeW_label(userNum, relationFileName, label, args.diagnol, args.showheatmap)   # Generate user relation matrix
        GW = initializeGW_label(Gepsilon,userNum, relationFileName, label, args.diagnol)            
    else:
        normalizedNewW, newW, label = initializeW_clustering(OriginaluserNum, relationFileName, nClusters)
        GW = initializeGW_clustering(Gepsilon, relationFileName, newW)
        W = normalizedNewW
    # Decide which algorithms to run.
    runCoLinUCB = runGOBLin = runLinUCB = False
    if args.alg:
        if args.alg == 'CoLinUCB':
            runCoLinUCB = True
        elif args.alg == 'GOBLin':
            runGOBLin = True
        elif args.alg == 'LinUCB':
            runLinUCB = True
        elif args.alg == 'ALL':
            runColinUCB = runGOBLin = runLinUCB = True
    else:
        runColinUCB = runGOBLin = runLinUCB = True

    fileSig = args.dataset+'_'+str(nClusters)+'_shuffled_Clustering_'+args.alg+'_Diagnol_'+args.diagnol+'_'


    articles_random = randomStruct()
    CoLinUCB_USERS = CoLinUCBStruct(d, lambda_ ,userNum, W)
    GOBLin_USERS = GOBLinStruct(d, lambda_, userNum, GW)
    LinUCB_users = []  
    for i in range(OriginaluserNum):
        LinUCB_users.append(LinUCBStruct(d, lambda_ ))
     
    fileName = address + "/processed_events_shuffled.dat"
    fileNameWrite = os.path.join(save_address, fileSig + timeRun + '.csv')
    #FeatureVectorsFileName =  LastFM_address + '/Arm_FeatureVectors.dat'

    # put some new data in file for readability
    
    with open(fileNameWrite, 'a+') as f:
        f.write('\nNew Run at  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
        f.write('\n, Time,RandomReward; CoLinReward; LinUCBReward; GOBLinReward\n')

    print fileName, fileNameWrite

    with open(fileName, 'r') as f:
        f.readline()
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
           
            currentUserID =label[int(userID)] 

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
                    LinUCB_pta = LinUCB_users[int(userID)].getProb(alpha, article_featureVector)
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
            #if CoLinUCBPicked !=LinUCBPicked:
            #    print 'Error!!!!!'
            RandomPicked = choice(currentArticles)
            if RandomPicked == article_chosen:
                articles_random.reward +=1
            
            if CoLinUCBPicked == article_chosen:
                CoLinUCB_USERS.reward +=1
                CoLinReward = 1
            CoLinUCB_USERS.updateParameters(CoLinUCB_PickedfeatureVector,CoLinReward, currentUserID)
            
            if LinUCBPicked == article_chosen:
                LinUCB_users[int(userID)].reward +=1
                LinUCBReward = 1
            LinUCB_users[int(userID)].updateParameters(LinUCB_PickedfeatureVector, LinUCBReward)
            
        
            if GOBLinPicked == article_chosen:
                GOBLin_USERS.reward +=1
                GOBLinReward = 1
            GOBLin_USERS.updateParameters(GOBLin_PickedfeatureVector, GOBLinReward, currentUserID)

            
            # if the batch has ended
            if totalObservations%batchSize==0:
                printWrite()
    #print stuff to screen and save parameters to file when the Yahoo! dataset file ends
    printWrite()
    
