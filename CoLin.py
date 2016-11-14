import numpy as np
from util_functions import vectorize, matrixize


class CoLinUCBUserSharedStruct(object):
    def __init__(self, featureDimension, lambda_, userNum, W, RankoneInverse):
        self.W = W
        self.userNum = userNum
        self.d = int(featureDimension)
        self.A = lambda_ * np.identity(n=featureDimension * userNum)  # accumulated feature matrix, a dN by dN matrix
        self.CCA = np.identity(n=featureDimension * userNum)  # inverse of A, a dN by dN matrix
        self.b = np.zeros(featureDimension * userNum)

        self.AInv = np.linalg.inv(self.A)

        self.UserTheta = np.zeros(shape=(featureDimension, userNum))
        self.CoTheta = np.zeros(shape=(featureDimension, userNum))

        self.BigW = np.kron(np.transpose(W), np.identity(n=featureDimension))
        self.RankoneInverse = RankoneInverse

    def updateParameters(self, articlePicked, click, userID):
        X = vectorize(np.outer(articlePicked, self.W.T[userID]))

        self.A += np.outer(X, X)
        self.b += click * X
        # fout=open('CoLinPower','a+')
        # fout.write(str(self.A)+'\n')
        # fout.close()
        if self.RankoneInverse:
            temp = np.dot(self.AInv, X)
            self.AInv = self.AInv - (np.outer(temp, temp)) / (1.0 + np.dot(np.transpose(X), temp))
        else:
            self.AInv = np.linalg.inv(self.A)

        self.UserTheta = matrixize(np.dot(self.AInv, self.b), len(articlePicked))
        self.CoTheta = np.dot(self.UserTheta, self.W)
        self.CCA = np.dot(np.dot(self.BigW, self.AInv), np.transpose(self.BigW))

    def getProb(self, alpha, articleFeatureVector, userID):
        TempFeatureV = np.zeros(len(articleFeatureVector) * self.userNum)
        TempFeatureV[int(userID) * self.d:(int(userID) + 1) * self.d] = np.asarray(articleFeatureVector)
        # print self.CoTheta.T[userID]
        # print articleFeatureVector
        np.dot(self.CoTheta.T[userID], self.CoTheta.T[userID])
        np.dot(articleFeatureVector, articleFeatureVector)
        mean = np.dot(self.CoTheta.T[userID], articleFeatureVector)
        var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))
        pta = mean + alpha * var
        # return round(pta,12)
        return pta, mean, var


# ---------------CoLinUCB(fixed user order) algorithms: Asynisized version and Synchorized version
class CoLinUCBAlgorithm:
    def __init__(self, dimension, alpha, lambda_, n, W, RankoneInverse=False):  # n is number of users
        self.USERS = CoLinUCBUserSharedStruct(dimension, lambda_, n, W, RankoneInverse)
        self.dimension = dimension
        self.alpha = alpha
        self.W = W

        self.CanEstimateUserPreference = True
        self.CanEstimateCoUserPreference = True
        self.CanEstimateW = False

    def decide(self, pool_articles, userID):
        maxPTA = float('-inf')
        articlePicked = None

        for x in pool_articles:
            x_pta = self.USERS.getProb(self.alpha, x.featureVector, userID)
            # pick article with highest Prob
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta

        return articlePicked

    def updateParameters(self, articlePicked, click, userID):
        self.USERS.updateParameters(articlePicked.featureVector, click, userID)

    def getTheta(self, userID):
        return self.USERS.UserTheta.T[userID]

    def getCoTheta(self, userID):
        return self.USERS.CoTheta.T[userID]

    def getA(self):
        return self.USERS.A


# -----------CoLinUCB select user algorithm(only has asynchorize version)-----
class CoLinUCB_SelectUserAlgorithm(CoLinUCBAlgorithm):
    def __init__(self, dimension, alpha, lambda_, n, W, RankoneInverse=False):  # n is number of users
        CoLinUCBAlgorithm.__init__(self, dimension=dimension, alpha=alpha, lambda_=lambda_, n=n, W=W,
                                   RankoneInverse=RankoneInverse)
        self.USERS = CoLinUCBUserSharedStruct(dimension, lambda_, n, W, RankoneInverse)

    def decide(self, pool_articles, AllUsers):
        maxPTA = float('-inf')
        articlePicked = None
        userPicked = None

        for x in pool_articles:
            for user in AllUsers:
                x_pta = self.USERS.getProb(self.alpha, x.featureVector, user.id)
                # pick article with highest Prob
                if maxPTA < x_pta:
                    articlePicked = x
                    userPicked = user
                    maxPTA = x_pta

        return userPicked, articlePicked
