import time
import sys
import os
import numpy as np
from sklearn import cluster
def initializeW_clustering(n,relationFileName):
    W = np.identity(n+1)
    with open(relationFileName) as f:
        f.readline()
        for line in f:
            line = line.split('\t')            
            if int(line[0])<=n and int(line[1]) <=n:
                W[int(line[0])][int(line[1])] +=1 
    return W

relationFileName = sys.argv[1]
save_path = sys.argv[2]
n = 1900
W = initializeW_clustering(n, relationFileName)



ms = cluster.MeanShift()
clustering_names = ['MeanShift']
clustering_algorithms = [ms]

for nClusters in [50, 100, 200]:
    spc = cluster.SpectralClustering(n_clusters=nClusters, affinity="nearest_neighbors")
    clustering_names.append('SpectralClustering'+str(nClusters))
    clustering_algorithms.append(spc)

for nClusters in [50, 100, 200]:
    kmeans = cluster.KMeans(n_clusters=nClusters)
    clustering_names.append('KMeans'+str(nClusters))
    clustering_algorithms.append(kmeans)

for name, algorithm in zip(clustering_names, clustering_algorithms):
        print ("Start "+name)
        t0 = time.time()
        algorithm.fit(W)
        t1 = time.time()
        print (name, t1-t0)
        label = algorithm.labels_
        with open(os.path.join(path, name)+'.cluster','w') as f:
            for i in range(len(label)):
                f.write(str(label[i])+'\n')

