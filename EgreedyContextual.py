import numpy as np
import random
from sklearn.preprocessing import normalize
class EgreedyContextualSharedStruct:
    def __init__(self, Tu, m, lambd, alpha, userNum, itemNum,k, feature_dim, tau=0.05, r=0, init='zero'):
        self.reward = 0
        # self.R = np.zeros((userNum, itemNum))
        self.S = np.zeros((userNum, itemNum))
        self.time = 1
        self.tau = tau #SGD Learning rate
        self.Tu = Tu
        self.m =m
        self.lambd = lambd
        self.alpha = alpha 
        if (init == 'random'):
            self.U = np.random.rand(userNum,k)
            self.V = np.random.rand(k,itemNum)
        else:
            self.U = np.zeros((userNum,k))
            self.V = np.zeros((k,itemNum))

        self.feature_dim = feature_dim
        self.r = r

        #add normalization
        self.U = normalize(self.U, axis=1, norm='l1')
        self.V = normalize(self.V, axis=0, norm='l1')
        self.k = k
    def choose_arm(self, items, userID):
        max_r = float('-inf')
        max_itemID = None
        #print (items, userID)
        for itemID in items:
            self.V[:self.feature_dim, itemID] = self.features[itemID]
            restimate = self.U[userID].dot(self.V[:, itemID])
            if (max_r < restimate):
                max_r = restimate
                max_itemID = itemID
        #print (max_r, max_itemID)
        epsilon = self.get_epsilon()
        if random.random() > epsilon:
            return max_itemID
        else:
            return random.choice(items) 
    def getProb(self, itemID, userID, features):
        self.V[:self.feature_dim, itemID] = features
        return self.U[userID].dot(self.V[:, itemID])

    def updateParameters(self, reward, itemID, userID, features):
        self.time += 1
        #self.S[userID, itemID] = (reward+1)
        #self.mBALS_WR(self.U,self.V,self.S,self.lambd,1,userID,itemID,self.k)
        self.SGD(reward, itemID, userID, self.feature_dim, features)
        #self.SGD_L2(reward, itemID, userID)
    def get_epsilon(self):
        return min(self.alpha/self.time, 1)

    def SGD(self, reward, itemID, userID, feature_dim, features):
        restimate = self.U[userID].dot(self.V[:, itemID])
        self.U[userID] += 2*self.tau*(reward-restimate)*self.V[:, itemID]
        self.V[:, itemID] += 2*self.tau*(reward-restimate)*self.U[userID]
        self.V[:feature_dim, itemID] = features
    def SGD_L2(self, reward, itemID, userID, feature_dim):
        restimate = self.U[userID].dot(self.V[:, itemID])
        self.U[userID] += 2*self.tau*((reward-restimate)*self.V[:, itemID] - self.r*self.U[userID])
        self.V[:, itemID] += 2*self.tau*((reward-restimate)*self.U[userID] - self.r*self.V[:, itemID])
    def mBALS_WR(self, U,V,S,lambd,m,i_t,j_t,k,tk=0):
        m1,n1=S.shape
        temp1=range(m1-1)
        Wi=random.sample(temp1,m-1)
        for i in range(len(Wi)):
            if Wi[i]>=i_t:
                Wi[i]+=1
        Wi.append(i_t)
        temp1=range(n1-1)
        Wj=random.sample(temp1,m-1)
        for i in range(len(Wj)):
            if Wj[i]>=j_t:
                Wj[i]+=1
        Wj.append(j_t)
        #print Wi,Wj
        W = (S>1e-9)+.0 # rated = 1 , unrated = 0
        counti=np.sum(W,axis=1)
        countj=np.sum(W,axis=0)
        
        for u in Wi:
            Wu=W[u]
            V1=V.copy()
            flag=0
            for i,wi in enumerate(Wu):
                if wi==0:
                    V1[:,i]=np.zeros(k)
                else:
                    flag=1
            if (flag!=0):
                U[u] = np.linalg.solve(np.dot(V1,V1.T)+lambd*counti[u]*np.eye(k),np.dot(V1,S[u].T)).T
        for v in Wj:
            Wv=W.T[v]
            U1=U.copy()
            flag=0
            for i,wi in enumerate(Wv):
                if wi==0:
                    U1[i]=np.zeros(k)
                else:
                    flag=1
            if (flag!=0):
                V[:,v][tk:] = np.linalg.solve(np.dot(U1.T,U1)+lambd*countj[v]*np.eye(k),np.dot(U1.T,S[:,v]))[tk:]
        return U,V


