import sys
import numpy as np
from YahooExp_util_functions import *

num = sys.argv[2]
userFeatureVectors = getClusters('YahooKMeansModel/10kmeans_model' + str(num) + '.dat')

yahooData_address = sys.argv[1]
dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
for dataDay in dataDays:
    fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay + '.userID.statistic'
    fin = open(fileName, 'r')
    fout = open(fileName + '.userID', 'w')
    for line in fin:
        line = line.split("|")
        currentUser_featureVector = np.array([float(x.strip().split(':')[1]) for x in line[1].strip().split(' ')[1:]])[
                                    :-1]
        currentUserID = getIDAssignment(np.asarray(currentUser_featureVector), userFeatureVectors)
        line[1] = str(currentUserID)
        fout.write('|'.join(line))
fout.close()
