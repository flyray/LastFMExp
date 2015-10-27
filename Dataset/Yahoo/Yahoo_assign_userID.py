import sys
import numpy as np
from YahooExp_util_functions import *
userFeatureVectors = getClusters('YahooKMeansModel/10kmeans_model160.dat')  



fin = open(sys.argv[1], 'r')
fout = open(sys.argv[1]+'.userID','w')
for line in fin:
    line = line.split("|")
    currentUser_featureVector = np.array([float(x.strip().split(':')[1]) for x in line[1].strip().split(' ')[1:]])[:-1]
    currentUserID = getIDAssignment(np.asarray(currentUser_featureVector), userFeatureVectors)
    line[1] = str(currentUserID)
    fout.write('|'.join(line))
    

