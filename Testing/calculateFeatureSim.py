# coding = utf-8
import numpy as np

def readFeatureVectors(FeatureVectorsFileName):
    FeatureVectors = {}
    with open(FeatureVectorsFileName, 'r') as f:
        f.readline()
        for line in f:
            line = line.split("\t")
            vec = line[1].strip('[]').strip('\n').split(';')
            FeatureVectors[int(line[0])] = np.array(vec)
    return FeatureVectors


def saveToFile(fileNameWrite, featureVectors):
    with open(fileNameWrite, 'a+') as f:
        for i in featureVectors.keys():
            f.write(str(i))
            f.write('\t')
            f.write((featureVectors[i]))
            f.write('\n')

        f.close()


def calculateDis(pastFeature, currentFeature):
    distance = 0
    for i in range(len(pastFeature)):
        distance = distance + np.sqrt(np.square(float(pastFeature[i]) - float(currentFeature[i])))
    return distance


def saveDisToFile(fileNameWrite, disList):
    with open(fileNameWrite, 'a+') as f:
        f.write('0')
        f.write('\t')
        for i in range(len(disList)):
            f.write(str(i))
            f.write('\t')
        f.write('\n')
        for i in range(len(disList)):
            print 'writing: ', i
            f.write(str(i))
            f.write('\t')
            for j in range(len(disList[i])):
                f.write(str(disList[i][j]))
                f.write('\t')
            f.write('\n')
        f.close()


if __name__ == '__main__':
    featureVectors = {}
    writeFilePath = '../Dataset/hetrec2011-lastfm-2k/Arm_FeatureVectors_WithOrder.dat'
    writeDisFilePath = '../Dataset/hetrec2011-lastfm-2k/Arm_FeatureVectors_Dis.dat'
    fileName = '../Dataset/hetrec2011-lastfm-2k/Arm_FeatureVectors_2.dat'

    featureVectors = readFeatureVectors(fileName)

    resultList = [[0 for i in range(18746)] for i in range(18746)]  # max 18744, 18746

    print "start writing"
    for i in featureVectors.keys():
        print 'calcuateDis: ', i
        for j in featureVectors.keys():
            distance = calculateDis(featureVectors[i], featureVectors[j])
            resultList[i][j] = distance

    saveDisToFile(writeDisFilePath, resultList)