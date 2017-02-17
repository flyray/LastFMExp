# coding=utf-8
from conf import *  # it saves the address of data stored and where to save the data produced by algorithms
import time
from LastFM_util_functions_simple import *
from LinUCB import LinUCBUserStruct
import datetime

import numpy
import theano
import theano.tensor as T


class LastFMStruct(object):

    def __init__(self, n_in, n_out):

        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
    def __init__(self):
        self.reward = 0


# structure to save data from LinUCB strategy
class LinUCBStruct(LinUCBUserStruct):
    def __init__(self, featureDimension, lambda_, RankoneInverse=False):
        LinUCBUserStruct.__init__(self, featureDimension=featureDimension, lambda_=lambda_,
                                  RankoneInverse=RankoneInverse)
        self.reward = 0


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

if __name__ == '__main__':
    startTime = time.clock()
    startT = datetime.datetime.now()
    startT.strftime('%Y-%m-%d %H:%M:%S')
    print "start running!"

    def printWrite():
        LinUCBTotalReward = 0
        for i in range(OriginaluserNum):
            LinUCBTotalReward += LinUCB_users[i].reward

        recordedStats = [articles_random.reward]
        s = 'random ' + str(articles_random.reward)

        s += '  LinUCB ' + str(LinUCBPicked) + ' ' + str(LinUCBTotalReward)
        recordedStats.append(LinUCBPicked)
        recordedStats.append(LinUCBTotalReward)

        # print s, write to file
        save_to_file(fileNameWrite, recordedStats, tim)


    timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S')  # the current data time

    # 需要初始化的数据
    diagnol = 'Origin'
    RankoneInverse = False
    dataset = 'LastFM'

    batchSize = 1  # size of one batch
    d = 25  # feature dimension
    alpha = 0.3  # control how much to explore
    lambda_ = 0.2  # regularization used in matrix A
    Gepsilon = 0.3  # Parameter in initializing GW

    totalObservations = 0

    OriginaluserNum = 2100
    nClusters = 100
    userNum = nClusters

    # 所用数据集
    relationFileName = LastFM_relationFileName  # user_friends.dat.mapped
    address = LastFM_address  # ./Dataset/hetrec2011-lastfm-2k
    FeatureVectorsFileName = LastFM_FeatureVectorsFileName  # Arm_FeatureVectors_2.dat
    save_address = LastFM_save_address  # ./LastFMResults

    normalizedNewW, newW, label = initializeW_clustering(OriginaluserNum, relationFileName, nClusters)

    # Read Feature Vectors from File
    FeatureVectors = readFeatureVectorFile(FeatureVectorsFileName)

    # 初始化算法和数据
    runLinUCB = True
    fileName = address + "/processed_events_shuffled.dat"
    fileSig = dataset  # 修改文件名,便于实验

    articles_random = randomStruct()

    LinUCB_users = []
    for i in range(OriginaluserNum):
        LinUCB_users.append(LinUCBStruct(d, lambda_, RankoneInverse))

    # 保存数据地址
    fileNameWrite = os.path.join(save_address, fileSig + timeRun + '.csv')
    # FeatureVectorsFileName =  LastFM_address + '/Arm_FeatureVectors.dat'

    with open(fileNameWrite, 'a+') as f:
        f.write('\nNew Run at  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
        f.write('\n, Time, RandomReward; ')
        f.write('LinUCBReward; ')
        f.write('\n')

    print fileName, fileNameWrite

    tsave = 60 * 60 * 46  # Time interval for saving model is one hour.
    tstart = time.time()
    save_flag = 0
    printCount = 0
    with open(fileName, 'r') as f:  # processed_events_shuffled.dat
        f.readline()
        LinUCBTotalReward = 0
        # reading file line ie observations running one at a time
        for i, line in enumerate(f, 1):
            LinUCBReward = 0

            totalObservations += 1
            userID, tim, pool_articles = parseLine(line)
            currentArticles = []
            article_featureMatrix = []

            LinUCB_maxPTA = float('-inf')
            LinUCBPicked = None

            currentUserID = label[int(userID)]
            article_chosen = int(pool_articles[0])
            # for article in np.random.permutation(pool_articles) :

            for article in pool_articles:  # 对article pool中的文章进行遍历
                article_id = int(article.strip(']'))
                article_featureVector = FeatureVectors[article_id]
                article_featureVector = np.array(article_featureVector, dtype=float)
                article_featureMatrix.append(article_featureVector)
                currentArticles.append(article_id)

            LinUCB_pta = LinUCB_users[int(userID)].getMatrixProb(alpha, article_featureMatrix)
            maxPTA = np.max(LinUCB_pta)
            index_matPTA = numpy.argmax(LinUCB_pta)

            LinUCBPicked = int((pool_articles[index_matPTA]).strip(']'))
            temp_pickedFeatureVector = FeatureVectors[LinUCBPicked]
            LinUCB_pickedFeatureVector = np.array(temp_pickedFeatureVector, dtype=float)

            RandomPicked = choice(currentArticles)
            if RandomPicked == article_chosen:
                articles_random.reward += 1

            if LinUCBPicked == article_chosen:
                LinUCB_users[int(userID)].reward += 1
                LinUCBReward = 1
            LinUCB_users[int(userID)].updateParameters(LinUCB_pickedFeatureVector, LinUCBReward)  # 原代码

            if printCount % 100 == 0:
                print 'calculate on going! printCount: ', printCount
            printCount += 1

            if totalObservations % batchSize == 0:
                printWrite()
                tend = time.time()

    # print stuff to screen and save parameters to file when the Yahoo! dataset file ends
    printWrite()
    endTime = time.clock()

    endT = datetime.datetime.now()
    endT.strftime('%Y-%m-%d %H:%M:%S')

    print "end! time: %f s" % (endTime - startTime)
    print "start time: ", startT, "    end time: ", endT

