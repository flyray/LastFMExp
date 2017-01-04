# coding=utf-8
# import matplotlib.pyplot as plt
import argparse  # For argument parsing
# import os.path
from LinUCB2 import LinUCBUserStruct2
from conf import *  # it saves the address of data stored and where to save the data produced by algorithms
import time
import re  # regular expression library
from random import random, choice  # for random strategy
from operator import itemgetter
import datetime
import numpy as np
from scipy.sparse import csgraph
# from scipy.spatial import distance
# from YahooExp_util_functions import getClusters, getIDAssignment, parseLine, save_to_file, initializeW, vectorize, matrixize, articleAccess
from LastFM_util_functions_2 import *  # getFeatureVector, initializeW, initializeGW, parseLine, save_to_file, initializeW_clustering, initializeGW_clustering
# from LastFM_util_functions import getFeatureVector, initializeW, initializeGW, parseLine, save_to_file

from CoLin import CoLinUCBUserSharedStruct, CoLinUCBAlgorithm
from LinUCB import LinUCBUserStruct, Hybrid_LinUCBUserStruct
from GOBLin import GOBLinSharedStruct
from W_Alg import *

import datetime

# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
    def __init__(self):
        self.reward = 0

# structure to save data from LinUCB strategy
class LinUCBStruct(LinUCBUserStruct2):
    def __init__(self, featureDimension, lambda_, RankoneInverse=False):
        LinUCBUserStruct2.__init__(self, featureDimension=featureDimension, lambda_=lambda_,
                                  RankoneInverse=RankoneInverse)
        self.reward = 0


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


# 读取已经计算好的向量距离矩阵
def readFeatureDis(filePath):
    print 'Strat read feature distance matrix ! \n reading...'
    featureDis = []
    with open(filePath, 'r') as f:
        f.readline()
        for line in f:
            if float(line[0]) > 0:
                tempLine = line.split("\t")
                del tempLine[0]
                del tempLine[0]
                featureDis.append(tempLine)
    f.close()
    print "Read feature distance matrix over!"
    return featureDis


if __name__ == '__main__':

    startTime = time.clock()
    startT = datetime.datetime.now()
    startT.strftime('%Y-%m-%d %H:%M:%S')
    print "start running!"

    # 读取向量距离矩阵
    # featureDisM = readFeatureDis(featureDistanceMatrix)

    def printWrite():
        if runLinUCB:
            LinUCBTotalReward = 0
            for i in range(OriginaluserNum):
                LinUCBTotalReward += LinUCB_users[i].reward

                # print totalObservations
        recordedStats = [articles_random.reward]
        s = 'random ' + str(articles_random.reward)

        if runLinUCB:
            s += '  LinUCB ' + str(LinUCBPicked) + ' ' + str(LinUCBTotalReward)
            recordedStats.append(LinUCBPicked)
            recordedStats.append(LinUCBTotalReward)

        save_to_file(fileNameWrite, recordedStats, tim)


    timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S')  # the current data time

    # Introduce argparse for future argument parsing.
    parser = argparse.ArgumentParser(description='')

    # If exsiting cluster label data.
    parser.add_argument('--clusterfile', dest="clusterfile", help="input an clustering label file",
                        metavar="FILE", type=lambda x: is_valid_file(parser, x))
    # Select algorithm.
    parser.add_argument('--alg', dest='alg',
                        help='Select a specific algorithm, could be CoLinUCB, GOBLin, LinUCB, M_LinUCB, Uniform_LinUCB, Hybrid_LinUCB, LearnW, or ALL. No alg argument means Random.')

    # Designate relation matrix diagnol.
    parser.add_argument('--diagnol', dest='diagnol', required=True,
                        help='Designate relation matrix diagnol, could be 0, 1, or Origin.')
    # Whether show heatmap of relation matrix.
    parser.add_argument('--showheatmap', action='store_true',
                        help='Show heatmap of relation matrix.')

    parser.add_argument('--RankoneInverse', action='store_true',
                        help='Use Rankone Correction to do matrix inverse.')
    parser.add_argument('--WRegu', action='store_true',
                        help='Regularization to true W.')
    # Dataset.
    parser.add_argument('--dataset', required=True, choices=['LastFM', 'Delicious'],
                        help='Select Dataset to run, could be LastFM or Delicious.')

    # Load previous running status. Haven't finished.
    parser.add_argument('--load',
                        help='Load previous running status. Such as Delicious_200_shuffled_Clustering_GOBLin_Diagnol_Opt__09_30_15_23_17 .')

    # Stop at certain line number. Haven't finished.
    parser.add_argument('--line', type=int,
                        help='Stop at certain line number, debug use.')

    # Designate event file, default is processed_events_shuffled.dat
    parser.add_argument('--event',
                        help='Designate event file. Default is processed_events_shuffled.dat')
    args = parser.parse_args()

    batchSize = 1  # size of one batch

    d = 25  # feature dimension
    alpha = 0.3  # control how much to explore
    lambda_ = 0.2  # regularization used in matrix A
    Gepsilon = 0.3  # Parameter in initializing GW

    totalObservations = 0
    RankoneInverse = args.RankoneInverse
    WRegu = args.WRegu

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
        userNum = nClusters = int(args.clusterfile.name.split('.')[-1])  # Get cluster number.
        W = initializeW_label(userNum, relationFileName, label, args.diagnol,
                              args.showheatmap)  # Generate user relation matrix
        GW = initializeGW_label(Gepsilon, userNum, relationFileName, label, args.diagnol)
    else:
        # LinUCB方法不需要下面这三句,加上影响速度,去掉
        normalizedNewW, newW, label = initializeW_clustering(OriginaluserNum, relationFileName, nClusters)
        GW = initializeGW_clustering(Gepsilon, relationFileName, newW)
        W = normalizedNewW
        # print "LinUCB method, cut W"
    # Read Feature Vectors from File
    FeatureVectors = readFeatureVectorFile(FeatureVectorsFileName)
    # Generate user feature vectors

    userFeatureVectors = generateUserFeature(W)
    # Decide which algorithms to run.
    runLinUCB = False
    if args.alg:
        if args.alg == 'LinUCB':
            runLinUCB = True
        elif args.alg == 'ALL':
            runCoLinUCB = runLearnW = runGOBLin = runLinUCB = run_M_LinUCB = run_Uniform_LinUCB = run_Hybrid_LinUCB = True
    else:
        args.alg = 'Random'
        # runCoLinUCB = runGOBLin = runLinUCB = run_M_LinUCB = run_Uniform_LinUCB= True

    if (args.event):
        fileName = args.event
    else:
        fileName = address + "/processed_events_shuffled.dat"

    # fileSig = args.dataset + '_' + args.clusterfile.name.split('/')[-1] + '_shuffled_Clustering_' + args.alg + '_Diagnol_' + args.diagnol + '_' + fileName.split('/')[3] + '_IniW2000'
    # fileSig = args.dataset + '_'  + '_shuffled_Clustering_' + args.alg + '_Diagnol_' + args.diagnol + '_' + fileName.split('/')[3] + '_IniW2000'
    fileSig = args.dataset  # 修改文件名,便于实验

    articles_random = randomStruct()
    if args.load:
        fileSig, timeRun = args.load.split('__')
        fileSig = fileSig + '_'
        timeRun = '_' + timeRun
        print fileSig, timeRun
        with open(args.load + '.model', 'r') as fin:
            obj = pickle.load(fin)
        with open(args.load + '.txt', 'r') as fin:
            finished_line = int(fin.readline().strip().split()[1])
            print finished_line
        if runLinUCB:
            LinUCB_users = obj
    else:
        if runLinUCB:
            LinUCB_users = []
            for i in range(OriginaluserNum):
                LinUCB_users.append(LinUCBStruct(d, lambda_, RankoneInverse))

    fileNameWrite = os.path.join(save_address, fileSig + timeRun + '.csv')
    # FeatureVectorsFileName =  LastFM_address + '/Arm_FeatureVectors.dat'

    # put some new data in file for readability

    with open(fileNameWrite, 'a+') as f:
        f.write('\nNew Run at  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
        f.write('\n, Time, RandomReward; ')
        if runLinUCB:
            f.write('LinUCBReward; ')
        f.write('\n')

    print fileName, fileNameWrite

    tsave = 60 * 60 * 46  # Time interval for saving model is one hour.
    tstart = time.time()
    save_flag = 0
    printCount = 0
    with open(fileName, 'r') as f:
        f.readline()
        if runLinUCB:
            LinUCBTotalReward = 0
        # reading file line ie observations running one at a time
        for i, line in enumerate(f, 1):
            if args.load:
                if i < finished_line:
                    continue
            if runLinUCB:
                LinUCBReward = 0
            totalObservations += 1
            userID, tim, pool_articles = parseLine(line)
            # tim, article_chosen, click, user_features, pool_articles = parseLine(line)
            # currentUser_featureVector = user_features[:-1]
            # currentUserID = getIDAssignment(np.asarray(currentUser_featureVector), userFeatureVectors)

            currentArticles = []

            if runLinUCB:
                LinUCB_maxPTA = float('-inf')
                LinUCBPicked = None

            currentUserID = label[int(userID)]
            article_chosen = int(pool_articles[0])
            # for article in np.random.permutation(pool_articles) :
            for article in pool_articles:
                article_id = int(article.strip(']'))
                # print article_id
                article_featureVector = FeatureVectors[article_id]
                article_featureVector = np.array(article_featureVector, dtype=float)
                # print article_featureVector
                currentArticles.append(article_id)
                # CoLinUCB pick article
                if len(article_featureVector) == 25:
                    if runLinUCB:
                        # 12.26添加内容
                        # LinUCB_users[int(userID)].calculateSim(article_id, featureDisM)
                        # LinUCB_users[int(userID)].calculateParameter()

                        LinUCB_pta = LinUCB_users[int(userID)].getProb(alpha, article_featureVector)
                        if LinUCB_maxPTA < LinUCB_pta:
                            LinUCBPicked = article_id
                            LinUCB_PickedfeatureVector = article_featureVector
                            LinUCB_PickedfeatureID = article_id
                            LinUCB_maxPTA = LinUCB_pta

            # article picked by random strategy
            # article_chosen = currentArticles[0]
            # print article_chosen, CoLinUCBPicked, LinUCBPicked, GOBLinPicked
            # if CoLinUCBPicked !=LinUCBPicked:
            #    print 'Error!!!!!'
            RandomPicked = choice(currentArticles)
            if RandomPicked == article_chosen:
                articles_random.reward += 1

            if runLinUCB:
                if LinUCBPicked == article_chosen:
                    LinUCB_users[int(userID)].reward += 1
                    LinUCBReward = 1
                LinUCB_users[int(userID)].updateParameters(LinUCB_PickedfeatureVector, LinUCBReward)  # 原代码

                # 1.2 存储状态
                # LinUCB_users[int(userID)].writeMemory(LinUCB_PickedfeatureVector, LinUCBReward, LinUCB_PickedfeatureID)
                if printCount % 100 == 0:
                    print 'calculate on going! printCount: ', printCount
                printCount += 1

                # 每次都用选中的arm更新
                # article_chosenFeature = FeatureVectors[article_chosen]
                # article_chosenFeature = np.array(article_chosenFeature, dtype=float)
                # LinUCB_users[int(userID)].updateParameters(article_chosenFeature, LinUCBReward)
                if save_flag:
                    print "Start saving model"
                    model_name = args.dataset + '_' + str(
                        nClusters) + '_shuffled_Clustering_LinUCB_Diagnol_' + args.diagnol + '_' + timeRun
                    model_dump(LinUCB_users, model_name, i)

            save_flag = 0
            # if the batch has ended
            if totalObservations % batchSize == 0:
                printWrite()
                tend = time.time()
                if tend - tstart > tsave:
                    save_flag = 1
                    tstart = tend
    # print stuff to screen and save parameters to file when the Yahoo! dataset file ends
    printWrite()
    endTime = time.clock()

    endT = datetime.datetime.now()
    endT.strftime('%Y-%m-%d %H:%M:%S')

    print "end! time: %f s" % (endTime - startTime)
    print "start time: ", startT, "    end time: ", endT

