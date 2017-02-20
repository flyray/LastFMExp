# coding=utf-8
import datetime
import numpy as np

LastFM_address = '../Dataset/hetrec2011-lastfm-2k'
fileName = LastFM_address + "/processed_events_shuffled.dat"
wholeData = LastFM_address + "/LastFMOrganizeData/wholeData.dat"


def parseLine(line):
    userID, tim, pool_articles = line.split("\t")
    userID, tim = int(userID), int(tim)
    pool_articles = np.array(pool_articles.strip('[').strip(']').strip('\n').split(','))
    return userID, tim, pool_articles


with open(fileName, 'r') as readData, open(wholeData, 'r+') as wholeDataWrite:

    j = 1
    wholeDataWrite.readline()
    for tempLine in wholeDataWrite:
        if j == 1:
            j += 1
            continue
        j += 1
        userID, tim, pool_articles = parseLine(tempLine)
        print 'userID: %s, time: %s, poolArticles %s' % (userID, tim, pool_articles)
