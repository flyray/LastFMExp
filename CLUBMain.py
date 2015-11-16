#import matplotlib.pyplot as plt
import argparse # For argument parsing
#import os.path
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
from CLUB import *

# structure to save data from random strategy as mentioned in LiHongs paper
class Article():    
    def __init__(self, id, FV=None):
        self.id = id
        self.featureVector = FV

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
        if runLinUCB:
            LinUCBTotalReward = 0
            for i in range(OriginaluserNum): 
                LinUCBTotalReward += LinUCB_users[i].reward
        if runCLUB:
            CLUBTotalReward = 0
            for i in range(OriginaluserNum): 
                CLUBTotalReward += CLUB.users[i].reward
        #print 'CULB'

        print totalObservations
        print 'random', articles_random.reward,'  CLUB', CLUBTotalReward

        #print totalObservations
        recordedStats = [articles_random.reward]
        s = 'random '+str(articles_random.reward)         
        if runLinUCB:
            s += '  LinUCB '+str(LinUCBPicked)+' '+str(LinUCBTotalReward)
            recordedStats.append(LinUCBPicked)
            recordedStats.append(LinUCBTotalReward)
        if runCLUB:
            s += '  CLUB '+str(CLUBPicked)+' '+str(CLUBTotalReward)
            recordedStats.append(CLUBPicked)
            recordedStats.append(CLUBTotalReward)
        #print s         
        # write to file
        save_to_file(fileNameWrite, recordedStats, tim) 


    timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S')  # the current data time
    
    # Introduce argparse for future argument parsing.
    parser = argparse.ArgumentParser(description='')

    # If exsiting cluster label data.
    parser.add_argument('--clusterfile', dest="clusterfile", help="input an clustering label file", 
                        metavar="FILE", type=lambda x: is_valid_file(parser, x))
    # Select algorithm.
    parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLinUCB, GOBLin, LinUCB, M_LinUCB, Uniform_LinUCB, or ALL. No alg argument means Random.')
   
    # Designate relation matrix diagnol.
    parser.add_argument('--diagnol', dest='diagnol', required=True,
                        help='Designate relation matrix diagnol, could be 0, 1, or Origin.') 
    # Whether show heatmap of relation matrix.
    parser.add_argument('--showheatmap', action='store_true',
                        help='Show heatmap of relation matrix.') 
    # Dataset.
    parser.add_argument('--dataset', required=True, choices=['LastFM', 'Delicious'],
                        help='Select Dataset to run, could be LastFM or Delicious.')

    # Load previous running status. Haven't finished.
    parser.add_argument('--load', 
                        help='Load previous running status. Such as Delicious_200_shuffled_Clustering_GOBLin_Diagnol_Opt__09_30_15_23_17 .')

    #Stop at certain line number. Haven't finished.
    parser.add_argument('--line', type=int,
                        help='Stop at certain line number, debug use.')

    # Designate event file, default is processed_events_shuffled.dat
    parser.add_argument('--event', 
                        help='Designate event file. Default is processed_events_shuffled.dat')    
    args = parser.parse_args()
    
    batchSize = 1                          # size of one batch
    
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
    # Read Feature Vectors from File
    FeatureVectors = readFeatureVectorFile(FeatureVectorsFileName)
    # Decide which algorithms to run.
    runCoLinUCB = runGOBLin = runLinUCB = run_M_LinUCB = run_Uniform_LinUCB= False
    if args.alg:
        if args.alg == 'CoLinUCB':
            runCoLinUCB = True
        elif args.alg == 'GOBLin':
            runGOBLin = True
        elif args.alg == 'LinUCB':
            runLinUCB = True
        elif args.alg =='M_LinUCB':
            run_M_LinUCB = True
        elif args.alg == 'Uniform_LinUCB':
            run_Uniform_LinUCB = True
        elif args.alg == 'CLUB':
            runCLUB =True
        elif args.alg == 'ALL':
            runCoLinUCB = runGOBLin = runLinUCB = run_M_LinUCB = run_Uniform_LinUCB=True
    else:
        args.alg = 'Random'
        #runCoLinUCB = runGOBLin = runLinUCB = run_M_LinUCB = run_Uniform_LinUCB= True
    
    if (args.event):
        fileName = args.event
    else:
        fileName = address + "/processed_events_shuffled.dat"
    
    fileSig = args.dataset+'_'+str(nClusters)+'_shuffled_Clustering_'+args.alg+'_Diagnol_'+args.diagnol+'_'+fileName.split('/')[3]+'_'


    articles_random = randomStruct()
    if args.load:
        fileSig, timeRun = args.load.split('__')
        fileSig = fileSig+'_'
        timeRun = '_'+timeRun
        print fileSig, timeRun        
        with open(args.load +'.model', 'r') as fin:
            obj = pickle.load(fin)
        with open(args.load +'.txt', 'r') as fin:
            finished_line = int(fin.readline().strip().split()[1])
            print finished_line
        if runLinUCB:
            LinUCB_users = obj
        if runCLUB:
            CLUBAlgorithm.users = obj
    else:
        
        if runLinUCB:
            LinUCB_users = []  
            for i in range(OriginaluserNum):
                LinUCB_users.append(LinUCBStruct(d, lambda_ ))
        if runCLUB:
            CLUB =CLUBAlgorithm(d,alpha, lambda_, OriginaluserNum)
    fileNameWrite = os.path.join(save_address, fileSig + timeRun + '.csv')
    #FeatureVectorsFileName =  LastFM_address + '/Arm_FeatureVectors.dat'

    # put some new data in file for readability
    
    with open(fileNameWrite, 'a+') as f:
        f.write('\nNew Run at  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
        f.write('\n, Time, RandomReward; ')
        if runCLUB:
            f.write('CLUBReward')
        f.write('\n')

    print fileName, fileNameWrite

    tsave = 60*60*46 # Time interval for saving model is one hour.
    tstart = time.time()
    save_flag = 0
    with open(fileName, 'r') as f:
        f.readline()
        if runLinUCB:
            LinUCBTotalReward  = 0
        # reading file line ie observations running one at a time
        for i, line in enumerate(f, 1):
            if args.load:
                if i< finished_line:
                    continue
            if runCLUB:
                CLUBReward = 0

            totalObservations +=1
            userID, tim, pool_articles = parseLine(line)  
            currentArticles = []
            article_chosen = int(pool_articles[0])  
            #for article in np.random.permutation(pool_articles) :
            ArticlePOOL = []
            for article in pool_articles:
                article_id = int(article.strip(']'))
                #print article_id
                article_featureVector = FeatureVectors[article_id]
                article_featureVector =np.array(article_featureVector ,dtype=float)
                currentArticles.append(article_id)
                if len(article_featureVector)==25:
                    ArticlePOOL.append(Article(article_id,article_featureVector))             
            if runCLUB:
                CLUB_PickedfeatureVector, CLUBPicked= CLUB.decide(ArticlePOOL,userID)
            RandomPicked = choice(currentArticles)
            if RandomPicked == article_chosen:
                articles_random.reward +=1
            if runCLUB:
                #print CLUBPicked, article_chosen
                if CLUBPicked == article_chosen:
                    CLUB.users[int(userID)].reward +=1
                    CLUBReward = 1
                CLUB.updateParameters(CLUB_PickedfeatureVector, CLUBReward,userID)
                CLUB.updateGraphClusters(userID)
                
                
            save_flag = 0
            # if the batch has ended
            if totalObservations%batchSize==0:
                printWrite()
                tend = time.time()
                if tend-tstart>tsave:
                    save_flag = 1
                    tstart = tend
    #print stuff to screen and save parameters to file when the Yahoo! dataset file ends
    printWrite()
    
