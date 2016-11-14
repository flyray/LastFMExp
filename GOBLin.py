import numpy as np
from scipy.linalg import sqrtm

from util_functions import vectorize, matrixize
from CoLin import CoLinUCBAlgorithm, CoLinUCB_SelectUserAlgorithm


class GOBLinSharedStruct:
    def __init__(self, featureDimension, lambda_, userNum, W, RankoneInverse):
        self.W = W
        self.userNum = userNum
        self.d = int(featureDimension)
        self.A = lambda_ * np.identity(n=self.d * userNum)
        self.b = np.zeros(featureDimension * userNum)
        self.AInv = np.linalg.inv(self.A)

        self.theta = np.dot(self.AInv, self.b)
        self.STBigWInv = sqrtm(np.linalg.inv(np.kron(W, np.identity(n=self.d))))
        self.STBigW = sqrtm(np.kron(W, np.identity(n=self.d)))
        self.RankoneInverse = RankoneInverse

    def updateParameters(self, articlePicked_FeatureVector, click, userID):
        # featureVectorM = np.zeros(shape = (self.d, self.userNum))
        featureVectorV = np.zeros(self.d * self.userNum)
        # featureVectorM.T[userID] = np.asarray(articlePicked_FeatureVector)
        # featureVectorV = vectorize(featureVectorM)
        featureVectorV[int(userID) * self.d: (int(userID) + 1) * self.d] = np.asarray(articlePicked_FeatureVector)

        CoFeaV = np.dot(self.STBigWInv, featureVectorV)
        self.A = self.A + np.outer(CoFeaV, CoFeaV)
        # print 'CoFeaVtype', type(CoFeaV), type(self.b), self.b.shape, CoFeaV.shape

        self.b = self.b + float(click) * CoFeaV

        if self.RankoneInverse:
            temp = np.dot(self.AInv, CoFeaV)
            self.AInv = self.AInv - (np.outer(temp, temp)) / (1.0 + np.dot(np.transpose(CoFeaV), temp))
        else:
            self.AInv = np.linalg.inv(self.A)

        self.theta = np.dot(self.AInv, self.b)

    def getProb(self, alpha, article_FeatureVector, userID):
        featureVectorV = np.zeros(self.d * self.userNum)
        featureVectorV[int(userID) * self.d:(int(userID) + 1) * self.d] = np.asarray(article_FeatureVector)

        CoFeaV = np.dot(self.STBigWInv, featureVectorV)

        mean = np.dot(np.transpose(self.theta), CoFeaV)
        var = np.sqrt(np.dot(np.dot(CoFeaV, self.AInv), CoFeaV))
        pta = mean + alpha * var
        # return round(pta,12)
        return pta


# inherite from CoLinUCBAlgorithm
class GOBLinAlgorithm(CoLinUCBAlgorithm):
    def __init__(self, dimension, alpha, lambda_, n, W, RankoneInverse=False):
        CoLinUCBAlgorithm.__init__(self, dimension=dimension, alpha=alpha, lambda_=lambda_, n=n, W=W,
                                   RankoneInverse=RankoneInverse)
        self.USERS = GOBLinSharedStruct(dimension, lambda_, n, W, RankoneInverse)

        self.CanEstimateUserPreference = False
        self.CanEstimateCoUserPreference = True
        self.CanEstimateW = False

    def getCoTheta(self, userID):
        thetaMatrix = matrixize(self.USERS.theta, self.dimension)
        return thetaMatrix.T[userID]


# inherite from CoLinUCB_SelectUserAlgorithm
class GOBLin_SelectUserAlgorithm(CoLinUCB_SelectUserAlgorithm):
    def __init__(self, dimension, alpha, lambda_, n, W, RankoneInverse=False):
        CoLinUCB_SelectUserAlgorithm.__init__(self, dimension=dimension, alpha=alpha, lambda_=lambda_, n=n, W=W,
                                              RankoneInverse=RankoneInverse)
        self.USERS = GOBLinSharedStruct(dimension, lambda_, n, W, RankoneInverse)
        self.CanEstimateUserPreference = False
        self.CanEstimateCoUserPreference = True
        self.CanEstimateW = False

    def getCoTheta(self, userID):
        thetaMatrix = matrixize(self.USERS.theta, self.dimension)
        return thetaMatrix.T[userID]
