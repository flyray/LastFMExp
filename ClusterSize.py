from conf import *  # it saves the address of data stored and where to save the data produced by algorithms
import argparse  # For argument parsing
import time
import re  # regular expression library
from random import random, choice  # for random strategy
from operator import itemgetter
import datetime
import numpy as np
import sys
from scipy.sparse import csgraph
from scipy.spatial import distance
from YahooExp_util_functions import *
import csv

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
        AsyCoLinUCBUserSharedStruct.__init__(self, featureDimension=featureDimension, lambda_=lambda_, userNum=userNum,
                                             W=W)
        self.learn_stats = articleAccess()


class GOBLinStruct(GOBLinSharedStruct):
    def __init__(self, featureDimension, lambda_, userNum, W):
        GOBLinSharedStruct.__init__(self, featureDimension=featureDimension, lambda_=lambda_, userNum=userNum, W=W)
        self.learn_stats = articleAccess()


class LinUCBStruct(LinUCBUserStruct):
    def __init__(self, featureDimension, lambda_):
        LinUCBUserStruct.__init__(self, featureDimension=featureDimension, lambda_=lambda_)
        self.learn_stats = articleAccess()


if __name__ == '__main__':
    # regularly print stuff to see if everything is going alright.
    # this function is inside main so that it shares variables with main and I dont wanna have large number of function arguments
    def printWrite():
        randomLearnCTR = articles_random.learn_stats.updateCTR()
        if algName == 'CoLin':
            CoLinUCBCTR = CoLinUCB_USERS.learn_stats.updateCTR()
            print totalObservations
            print 'random', randomLearnCTR, '  CoLin', CoLinUCBCTR
            recordedStats = [randomLearnCTR, CoLinUCBCTR]
        if algName == 'GOBLin':
            GOBLinCTR = GOBLin_USERS.learn_stats.updateCTR()
            print totalObservations
            print 'random', randomLearnCTR, '  GOBLin', GOBLinCTR
            recordedStats = [randomLearnCTR, GOBLinCTR]
        if algName == 'LinUCB':
            TotalLinUCBAccess = 0.0
            TotalLinUCBClick = 0.0
            for i in range(userNum):
                TotalLinUCBAccess += LinUCB_users[i].learn_stats.accesses
                TotalLinUCBClick += LinUCB_users[i].learn_stats.clicks

            if TotalLinUCBAccess != 0:
                LinUCBCTR = TotalLinUCBClick / (1.0 * TotalLinUCBAccess)
            else:
                LinUCBCTR = -1.0

            print totalObservations
            print 'random', randomLearnCTR, '	LinUCB', LinUCBCTR

            recordedStats = [randomLearnCTR, LinUCBCTR]

        # write to file
        save_to_file(fileNameWrite, recordedStats, tim)


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--YahooDataFile', dest="Yahoo_save_address", help="input the adress for Yahoo data")
    parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLin, GOBLin, LinUCB')

    parser.add_argument('--showheatmap', action='store_true',
                        help='Show heatmap of relation matrix.')
    parser.add_argument('--userNum', dest='userNum', help='Set the userNum, can be 20, 40, 80, 160')

    parser.add_argument('--Sparsity', dest='SparsityLevel',
                        help='Set the SparsityLevel by choosing the top M most connected users, should be smaller than userNum, when equal to userNum, we are using a full connected graph')
    parser.add_argument('--diag', dest="DiagType",
                        help="Specify the setting of diagional setting, can be set as 'Orgin' or 'Opt' ")

    args = parser.parse_args()

    algName = str(args.alg)
    clusterNum = int(args.userNum)
    SparsityLevel = int(args.SparsityLevel)
    yahooData_address = str(args.Yahoo_save_address)
    DiagType = str(args.DiagType)

    timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M')  # the current data time
    # dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    dataDays = ['01', '02']
    fileSig = 'Opt' + str(clusterNum) + 'SP' + str(SparsityLevel) + algName
    batchSize = 2000  # size of one batch

    d = 5  # feature dimension
    alpha = 0.3  # control how much to explore
    lambda_ = 0.2  # regularization used in matrix A
    epsilon = 0.3

    totalObservations = 0

    fileNameWriteCluster = os.path.join(Kmeansdata_address, '10kmeans_model' + str(clusterNum) + '.dat')
    userFeatureVectors = getClusters(fileNameWriteCluster)
    userNum = clusterNum
    if DiagType == 'Orgin':
        W = initializeW(userFeatureVectors, SparsityLevel)
    elif DiagType == 'Opt':
        W = initializeW_opt(userFeatureVectors, SparsityLevel)  # Generate user relation matrix
    GW = initializeGW(W, epsilon)

    articles_random = randomStruct()
    CoLinUCB_USERS = CoLinUCBStruct(d, lambda_, userNum, W)
    GOBLin_USERS = GOBLinStruct(d, lambda_, userNum, GW)
    LinUCB_users = []
    for i in range(userNum):
        LinUCB_users.append(LinUCBStruct(d, lambda_))
    totalID = {}

    for dataDay in dataDays:
        fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay
        fileNameWrite = os.path.join(Yahoo_save_address, fileSig + dataDay + timeRun + '.csv')

        # put some new data in file for readability

        print fileName, fileNameWrite
        with open(fileName, 'r') as f:
            # reading file line ie observations running one at a time
            for line in f:
                totalObservations += 1

                tim, article_chosen, click, user_features, pool_articles = parseLine(line)
                currentUser_featureVector = user_features[:-1]
                currentUserID = getIDAssignment(np.asarray(currentUser_featureVector), userFeatureVectors)
                if currentUserID not in totalID:
                    totalID[currentUserID] = 1
                else:
                    totalID[currentUserID] += 1
        print totalID
        sortedID = sorted(totalID.items(), key=itemgetter(1))

        w = csv.writer(open("./YahooResults/output.csv", "w"))
        for key, val in sortedID.items():
            w.writerow([key, val])
