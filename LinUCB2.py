import numpy as np
from YahooExp_util_functions import vectorize


def calculateDis(pastFeature, currentFeature):
    distance = 0
    for i in range(len(pastFeature)):
        distance = distance + np.sqrt(np.square(pastFeature[i] - currentFeature[i]))
    return distance


class LinUCBUserStruct2:
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
            # self.b += tempFeature * self.articleClickedList[i]
            self.b += tempFeature * self.articleClickedList[i] * self.simList[i]
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)

    def writeMemory(self, articlePicked_FeatureVector, click, articleId):
        self.articlePickedList.append(articlePicked_FeatureVector)
        self.articleClickedList.append(click)
        self.articleIdList.append(int(articleId))

    def calculateSimByFeature(self, pastFeature, currentFeature):
        distance = 0
        for i in range(len(pastFeature)):
            distance = distance + np.sqrt(np.square(float(pastFeature[i]) - float(currentFeature[i])))
        return distance

