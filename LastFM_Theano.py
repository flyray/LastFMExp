# coding=utf-8
from conf import *  # it saves the address of data stored and where to save the data produced by algorithms
import time
from LastFM_util_functions_simple import *
from LinUCB import LinUCBUserStruct
import datetime

import numpy
import theano
import theano.tensor as T


# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
    def __init__(self):
        self.reward = 0


# structure to save data from LinUCB strategy
class LinUCBStruct(LinUCBUserStruct):
    def __init__(self, featureDimension, lambda_, RankoneInverse=False, nIn=25, nOut=25):
        LinUCBUserStruct.__init__(self, featureDimension=featureDimension, lambda_=lambda_, RankoneInverse=RankoneInverse)

        self.reward = 0

        self.W = theano.shared(
            value=numpy.zeros(
                (nIn, nOut),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.Bias = theano.shared(
            value=numpy.zeros(
                (nOut,),
                dtype=theano.config.floatX
            ),
            name='Bias',
            borrow=True
        )

        # 计算加入权重矩阵之后的评分预估值
        self.estimateReward = []

        # 取出预估评分值中的最高值,作为预测item分值
        self.recommendReward = 0

        # parameters of the model
        self.params = [self.W, self.Bias]

        # keep track of model input
        # self.inputMean = inputMean
        # self.inputBias = inputBias

        # learningRate 我随机取的一个数字,应该还需要随后继续做调整
        self.learningRate = 0.13

    def calculateEstimateReward(self, inputMean, inputBias):
        self.estimateReward = T.nnet.softmax(T.dot(inputMean, self.W) + inputBias + self.Bias)

    def getRecommendReward(self):
        return T.argmax(self.estimateReward)

    # 定义loss function
    def costFunction(self):
        # 返回用户实际点击的那个Item数据
        return -T.log(self.estimateReward)[0, 0]

    def trainModel(self):
        gW = T.grad(cost=self.costFunction(), wrt=self.W)
        gb = T.grad(cost=self.costFunction(), wrt=self.Bias)

        updates = [(self.W, self.W - self.learningRate * gW),
                   (self.Bias, self.Bias - self.learningRate * gb)]

        train = theano.function(
            inputs=[],
            outputs=self.costFunction(),
            updates=updates
        )

        train()


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

    # normalizedNewW, newW, label = initializeW_clustering(OriginaluserNum, relationFileName, nClusters)

    # Read Feature Vectors from File
    FeatureVectors = readFeatureVectorFile(FeatureVectorsFileName)

    # 初始化算法和数据
    runLinUCB = True
    fileSig = dataset  # 修改文件名,便于实验
    # fileName = address + "/processed_events_shuffled.dat"

    # 测试加入代码正确性问题
    oneUserData = "/LastFMOrganizeData/oneUserData.dat"
    twoUserData = "/LastFMOrganizeData/twoUserData.dat"
    tenUserData = "/LastFMOrganizeData/tenUserData.dat"
    fileName = LastFM_address + oneUserData



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

            # currentUserID = label[int(userID)]
            article_chosen = int(pool_articles[0])
            # for article in np.random.permutation(pool_articles) :

            for article in pool_articles:  # 对article pool中的文章进行遍历
                article_id = int(article.strip(']'))
                article_featureVector = FeatureVectors[article_id]
                article_featureVector = np.array(article_featureVector, dtype=float)
                article_featureMatrix.append(article_featureVector)
                currentArticles.append(article_id)

            # getMatrixProb方法返回的数组分别是 [mean, var, linucb_pta]
            returnValues = LinUCB_users[int(userID)].getMatrixProb(alpha, article_featureMatrix)

            # 尝试去训练model
            tempMean = returnValues[0]
            tempVar = returnValues[1]
            LinUCB_users[int(userID)].calculateEstimateReward(tempMean, tempVar)
            LinUCB_users[int(userID)].trainModel()

            LinUCB_pta = returnValues[2]
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

