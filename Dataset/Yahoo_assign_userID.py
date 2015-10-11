import sys
import numpy as np
from YahooExp_util_functions import *
userFeatureVectors = getClusters('YahooKMeansModel/10kmeans_model160.dat')  

fin = open(sys.argv[1], 'r')
fout = open(sys.argv[1]+'.userID','w')
for line in fin:
    tim, articleID, click, user_features, pool_articles = parseLine(line)
    currentUser_featureVector = user_features[:-1]
    currentUserID = getIDAssignment(np.asarray(currentUser_featureVector), userFeatureVectors)
    fout.write(str(tim)+' '+str(articleID)+' '+str(click)+' '+str(currentUserID)+'\n')
    

