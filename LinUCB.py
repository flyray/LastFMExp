import numpy as np
from YahooExp_util_functions import vectorize


def calculateDis(pastFeature, currentFeature):
    distance = 0
    for i in range(len(pastFeature)):
        distance = distance + np.sqrt(np.square(pastFeature[i] - currentFeature[i]))
    return distance


class LinUCBUserStruct:
    def __init__(self, featureDimension, lambda_, RankoneInverse):
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.zeros(self.d)
        self.RankoneInverse = RankoneInverse

        #  add
        self.articlePickedList = []
        self.articleClickedList = []
        self.simList = []
        self.articleIdList = []

    def updateParameters(self, articlePicked_FeatureVector, click):
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector * click
        if self.RankoneInverse:
            temp = np.dot(self.AInv, articlePicked_FeatureVector)
            self.AInv = self.AInv - (np.outer(temp, temp)) / (
            1.0 + np.dot(np.transpose(articlePicked_FeatureVector), temp))
        else:
            self.AInv = np.linalg.inv(self.A)

        self.UserTheta = np.dot(self.AInv, self.b)

    def getTheta(self):
        return self.UserTheta

    def getA(self):
        return self.A

    def getProb(self, alpha, article_FeatureVector):
        mean = np.dot(self.UserTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
        pta = mean + alpha * var
        return pta

    def getMatrixProb(self, alpha, article_FeatureMatrix):
        mean = np.dot(self.UserTheta, np.transpose(article_FeatureMatrix))
        var = np.sqrt(np.diag(np.dot(np.dot(article_FeatureMatrix, self.AInv), np.transpose(article_FeatureMatrix))))
        ptaVector = mean + alpha * var
        return ptaVector

    def calculateSim(self, articleId, featureDisM):
        simList = []
        totalSim = 0.0
        # print 'articleIdList: ', self.articleIdList
        for i in range(len(self.articleIdList)):
            # print 'self.articleIdList[i]: ', self.articleIdList[i], 'articleId-1: ', articleId-1
            tempSim = float(featureDisM[self.articleIdList[i]][articleId-1])
            # print 'tempSim:', tempSim
            # print 'row :[][0][1]: ', featureDisM[self.articleIdList[i]+1][0], featureDisM[self.articleIdList[i]+2][1]
            simList.append(tempSim)
            totalSim += tempSim
        # print "totalSim: ", totalSim
        # print "simList: ", simList
        simList = map(lambda x: (totalSim+1) / (x+1), simList)
        self.simList = simList
        return simList

    def calculateParameter(self):
        for i in range(len(self.articlePickedList)):
            tempFeature = self.articlePickedList[i]
            self.A += np.outer(tempFeature, tempFeature)
            self.b += tempFeature * self.articleClickedList[i]
            # self.b += tempFeature * self.articleClickedList[i] * self.simList[i]
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)

    def writeMemory(self, articlePicked_FeatureVector, click, articleId):
        self.articlePickedList.append(articlePicked_FeatureVector)
        self.articleClickedList.append(click)
        self.articleIdList.append(int(articleId))


class Hybrid_LinUCB_singleUserStruct(LinUCBUserStruct):
    def __init__(self, userFeature, lambda_, RankoneInverse):
        LinUCBUserStruct.__init__(self, len(userFeature), lambda_, RankoneInverse)

        self.B = np.zeros([self.d, self.d ** 2])
        self.userFeature = userFeature

    def updateParameters(self, article_FeatureVector, click):
        additionalFeatureVector = vectorize(np.outer(self.userFeature, article_FeatureVector))
        LinUCBUserStruct.updateParameters(self, article_FeatureVector, click)
        self.B += np.outer(article_FeatureVector, additionalFeatureVector)

    def updateTheta(self, beta):
        self.UserTheta = np.dot(self.AInv, (self.b - np.dot(self.B, beta)))


class Hybrid_LinUCBUserStruct:
    def __init__(self, featureDimension, lambda_, userFeatureList, RankoneInverse=False):
        self.k = featureDimension ** 2
        self.A_z = lambda_ * np.identity(n=self.k)
        self.b_z = np.zeros(self.k)
        self.A_zInv = np.linalg.inv(self.A_z)
        self.beta = np.dot(self.A_zInv, self.b_z)
        self.users = []

        for i in range(len(userFeatureList)):
            self.users.append(Hybrid_LinUCB_singleUserStruct(userFeatureList[i], lambda_, RankoneInverse))

    def updateParameters(self, articlePicked_FeatureVector, click, userID):
        z = vectorize(np.outer(self.users[userID].userFeature, articlePicked_FeatureVector))

        temp = np.dot(np.transpose(self.users[userID].B), self.users[userID].AInv)

        self.A_z += np.dot(temp, self.users[userID].B)
        self.b_z += np.dot(temp, self.users[userID].b)

        self.users[userID].updateParameters(articlePicked_FeatureVector, click)

        temp = np.dot(np.transpose(self.users[userID].B), self.users[userID].AInv)

        self.A_z = self.A_z + np.outer(z, z) - np.dot(temp, self.users[userID].B)
        self.b_z = self.b_z + click * z - np.dot(temp, self.users[userID].b)
        self.A_zInv = np.linalg.inv(self.A_z)

        self.beta = np.dot(self.A_zInv, self.b_z)
        self.users[userID].updateTheta(self.beta)

    def getProb(self, alpha, article_FeatureVector, userID):
        x = article_FeatureVector
        z = vectorize(np.outer(self.users[userID].userFeature, article_FeatureVector))
        temp = np.dot(np.dot(np.dot(self.A_zInv, np.transpose(self.users[userID].B)), self.users[userID].AInv), x)
        mean = np.dot(self.users[userID].UserTheta, x) + np.dot(self.beta, z)
        s_t = np.dot(np.dot(z, self.A_zInv), z) + np.dot(np.dot(x, self.users[userID].AInv), x)
        -2 * np.dot(z, temp) + np.dot(np.dot(np.dot(x, self.users[userID].AInv), self.users[userID].B), temp)

        var = np.sqrt(s_t)
        pta = mean + alpha * var
        return pta


class Uniform_LinUCBAlgorithm(object):
    def __init__(self, dimension, alpha, lambda_, RankoneInverse=False):
        self.dimension = dimension
        self.alpha = alpha
        self.USER = LinUCBUserStruct(dimension, lambda_, RankoneInverse)

        self.CanEstimateUserPreference = False
        self.CanEstimateCoUserPreference = True
        self.CanEstimateW = False

    def decide(self, pool_articles, userID):
        maxPTA = float('-inf')
        articlePicked = None

        for x in pool_articles:
            x_pta = self.USER.getProb(self.alpha, x.featureVector)
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta
        return articlePicked

    def updateParameters(self, articlePicked, click, userID):
        self.USER.updateParameters(articlePicked.featureVector, click)

    def getCoTheta(self, userID):
        return self.USER.UserTheta


class Hybrid_LinUCBAlgorithm(object):
    def __init__(self, dimension, alpha, lambda_, userFeatureList, RankoneInverse=False):
        self.dimension = dimension
        self.alpha = alpha
        self.USER = Hybrid_LinUCBUserStruct(dimension, lambda_, userFeatureList, RankoneInverse)

        self.CanEstimateUserPreference = False
        self.CanEstimateCoUserPreference = True
        self.CanEstimateW = False

    def decide(self, pool_articles, userID):
        maxPTA = float('-inf')
        articlePicked = None

        for x in pool_articles:
            x_pta = self.USER.getProb(self.alpha, x.featureVector, userID)
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta
        return articlePicked

    def updateParameters(self, articlePicked, click, userID):
        self.USER.updateParameters(articlePicked.featureVector, click, userID)

    def getCoTheta(self, userID):
        return self.USER.users[userID].UserTheta


# ---------------LinUCB(fixed user order) algorithm---------------
class N_LinUCBAlgorithm:
    def __init__(self, dimension, alpha, lambda_, n, RankoneInverse=False):  # n is number of users
        self.users = []
        # algorithm have n users, each user has a user structure
        for i in range(n):
            self.users.append(LinUCBUserStruct(dimension, lambda_, RankoneInverse))

        self.dimension = dimension
        self.alpha = alpha

        self.CanEstimateUserPreference = False
        self.CanEstimateCoUserPreference = True
        self.CanEstimateW = False

    def decide(self, pool_articles, userID):
        maxPTA = float('-inf')
        articlePicked = None

        for x in pool_articles:
            x_pta = self.users[userID].getProb(self.alpha, x.featureVector)
            # pick article with highest Prob
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta

        return articlePicked

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.featureVector, click)

    def getCoTheta(self, userID):
        return self.users[userID].UserTheta


# -----------LinUCB select user algorithm-----------
class LinUCB_SelectUserAlgorithm(N_LinUCBAlgorithm):
    def __init__(self, dimension, alpha, lambda_, n, RankoneInverse=False):  # n is number of users
        N_LinUCBAlgorithm.__init__(self, dimension, alpha, lambda_, n, RankoneInverse)

    def decide(self, pool_articles, AllUsers):
        maxPTA = float('-inf')
        articlePicked = None
        userPicked = None

        for x in pool_articles:
            for user in AllUsers:
                x_pta = self.users[user.id].getProb(self.alpha, x.featureVector)
                # pick article with highest Prob
                if maxPTA < x_pta:
                    articlePicked = x
                    userPicked = user
                    maxPTA = x_pta

        return userPicked, articlePicked
