# coding=utf-8
import datetime
import numpy as np

LastFM_address = '../Dataset/hetrec2011-lastfm-2k'
fileName = LastFM_address + "/processed_events_shuffled.dat"
wholeData = LastFM_address + "/LastFMOrganizeData/wholeData.dat"
oneUserData = LastFM_address + "/LastFMOrganizeData/oneUserData.dat"


def parseLine(line):
    userID, tim, pool_articles = line.split("\t")
    userID, tim = int(userID), int(tim)
    pool_articles = np.array(pool_articles.strip('[').strip(']').strip('\n').split(','))
    return userID, tim, pool_articles

userList = []
eventList = []

# 声明100个空list用来保存 user event
for j in range(100):
    eventList.append([])


with open(fileName, 'r') as readData, open(wholeData, 'r+') as wholeDataWrite, open(fileName, 'r') as readData2:
    i = 0
    readData.readline()
    # 取前100个user
    for line in readData:
        userID, tim, pool_articles = parseLine(line)

        if not (userID in userList):
            userList.append(userID)
            i += 1
        if i == 100:
            break

    readData2.readline()
    for line2 in readData2:
        userID2, tim2, pool_articles2 = parseLine(line2)

        if userID2 in userList:
            tempIndex = userList.index(userID2)
            eventList[tempIndex].append(line2)

    for i in range(100):
        print 'eventList i = ', i, '  ;len():', len(eventList[i])
        for tempData in eventList[i]:
            wholeDataWrite.write(tempData)

        if i % 10 == 0:
            print "on going i:", i


# with open(fileName, 'r') as readData, open(wholeData, 'r+') as wholeDataWrite:
#     for line2 in readData:
#         userID2, tim2, pool_articles2 = parseLine(line2)
#
#         if userID2 in eventList:
#             tempIndex = eventList.index(userID2)
#             eventList[tempIndex].append(line2)
#
#     readData.close()
#
#     print '----  read all data OK ------'
#
#     for i in range(100):
#         for tempData in eventList[i]:
#             wholeDataWrite.write(tempData)
#
#         if i % 10 == 0:
#             print "on going i:", i

