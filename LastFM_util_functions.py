import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter
import numpy as np
from scipy.sparse import csgraph
from scipy.spatial import distance


def vectorize(M):
	temp = []
	for i in range(M.shape[0]*M.shape[1]):
		temp.append(M.T.item(i))
	V = np.asarray(temp)
	return V

def matrixize(V, C_dimension):
	temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	for i in range(len(V)/C_dimension):
		temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	W = temp
	return W

def getFeatureVector(FeatureVectorsFileName, articleID):
    FeatureVector = np.zeros(25)
    with open(FeatureVectorsFileName, 'r') as f:
        #print str(FeatureVectorsFileName)
        for line in f:

            line = line.split("\t")
            #print len(line)
            #print line[0]       
            if line[0] == str(articleID):
                FeatureVector = np.asarray(line[1].strip('[]').strip('\n').split(';'))
                #FeatureVector = FeatureVector.astype(np.float)
    return FeatureVector


# This code simply reads one line from the source files of Yahoo!
def parseLine(line):
        userID, tim, pool_articles = line.split("\t")
        userID, tim = int(userID), int(tim)
        pool_articles = np.array(pool_articles.strip('[').strip(']').strip('\n').split(','))
        #print pool_articles
      
        '''
        tim, articleID, click = line[0].strip().split("")
        tim, articleID, click = int(tim), int(articleID), int(click)
        user_features = np.array([float(x.strip().split(':')[1]) for x in line[1].strip().split(' ')[1:]])
        
        pool_articles = [l.strip().split(" ") for l in line[2:]]
        pool_articles = np.array([[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
        '''
        return userID, tim, pool_articles

def save_to_file(fileNameWrite, recordedStats, tim):
    with open(fileNameWrite, 'a+') as f:
        f.write('data') # the observation line starts with data;
        f.write(',' + str(tim))
        f.write(',' + ';'.join([str(x) for x in recordedStats]))
        f.write('\n')


def initializeGW( Gepsilon ,n, relationFileName):
    W = np.identity(n)
    with open(relationFileName) as f:
        for line in f:
            line = line.split('\t')
            if line[0] != 'userID':
                if int(line[0])<=n and int(line[1]) <=n:
                    W[int(line[0])][int(line[1])] +=1
    G = W
    L = csgraph.laplacian(G, normed = False)
    I = np.identity(n)
    GW = I + Gepsilon*L  # W is a double stochastic matrix
    #print GW          
    return GW.T

# generate graph W(No clustering)
def initializeW(n,relationFileName):
    W = np.identity(n)
    with open(relationFileName) as f:
        for line in f:
            line = line.split('\t')
            if line[0] != 'userID':
                if int(line[0])<=n and int(line[1]) <=n:
                    W[int(line[0])][int(line[1])] +=1
                    #print W[int(line[0])][int(line[1])]
    row_sums = W.sum(axis=1)
    NormalizedW = W / row_sums[:, np.newaxis]
    W = NormalizedW
    print W.T

    return W.T


