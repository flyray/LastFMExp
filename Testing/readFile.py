import time

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

if __name__ == '__main__':
    startTime = time.time()
    filePath = '../Dataset/hetrec2011-lastfm-2k/disTestData.txt'
    # filePath = '../Dataset/hetrec2011-lastfm-2k/Arm_FeatureVectors_Dis.dat'
    result = readFile(filePath, startTime)
    endTime = time.time()
    print result
    print 'total time: ', endTime - startTime
    print 'END!'