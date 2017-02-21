import time

import numpy as np

def parseLine(line):
    userID, tim, pool_articles = line.split("\t")
    userID, tim = int(userID), int(tim)
    pool_articles = np.array(pool_articles.strip('[').strip(']').strip('\n').split(','))
    # print pool_articles

    '''
    tim, articleID, click = line[0].strip().split("")
    tim, articleID, click = int(tim), int(articleID), int(click)
    user_features = np.array([float(x.strip().split(':')[1]) for x in line[1].strip().split(' ')[1:]])

    pool_articles = [l.strip().split(" ") for l in line[2:]]
    pool_articles = np.array([[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
    '''
    return userID, tim, pool_articles

def readFile(filePath, startTime):
    simList = []
    turn = 1
    with open(filePath, 'r') as f:
        f.readline()
        for line in f:
            # print line
            # print 'line[0]: ', line[0]
            if float(line[0]) > 0:
                tempLine = line.split("\t")
                # print 'tempLine: ', tempLine
                del tempLine[0]
                del tempLine[0]
                simList.append(tempLine)
    f.close()
    return simList


def readFiles(filePath):
    with open(filePath, 'r') as f:
        lines = f.readlines()
        # lines2 = f.readlines()
        i = 1
        for line in lines:
            if i % 15 == 0:
                print "Line:", i, "; value:", line
                userID, tim, pool_articles = parseLine(line)
                print userID, tim, pool_articles
            i += 1

    f.close()

if __name__ == '__main__':
    startTime = time.time()
    filePath = '../Dataset/hetrec2011-lastfm-2k/disTestData.txt'
    # filePath = '../Dataset/hetrec2011-lastfm-2k/Arm_FeatureVectors_Dis.dat'
    filePath2 = '../Dataset/hetrec2011-lastfm-2k/LastFMOrganizeData/oneUserData2.dat'
    # result = readFile(filePath, startTime)
    readFiles(filePath2)
    endTime = time.time()
    # print result
    print 'total time: ', endTime - startTime
    print 'END!'