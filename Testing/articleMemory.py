import numpy as np

data = [
    [1.5, 2.5, 3, 2, 4],
    [3, 5, 2, 4.5, 2.5],
    [1, 4.5, 3, 2, 1.5],
    [2.5, 2, 4, 3.5, 1],
    [3, 2.5, 3, 2, 1.5],
]

clickState = [1, 0, 1, 1, 1]

testArticle = [
    [2.5, 1.0, 4.5, 2.0, 3.0],
    [1.5, 2.0, 4.0, 3.5, 2.0]
]


def calculateDis(pastFeature, currentFeature):
    distance = 0
    for i in range(len(pastFeature)):
        distance = distance + np.sqrt(np.square(pastFeature[i] - currentFeature[i]))
    return distance

class ArticleStruct:
    def __init__(self):
        self.articlePickedList = []
        self.articleClickedList = []
        self.currentArticle = np.zeros(5)

    def calculateSim(self, currentFeature):
        simList = []
        totalSim = 0
        for i in range(len(self.articlePickedList)):
            tempSim = calculateDis(self.articlePickedList[i], currentFeature)
            simList.append(tempSim)
            totalSim += tempSim
        print "totalSim: ", totalSim
        print "simList: ", simList
        simList = map(lambda x: x / totalSim, simList)
        return simList

if __name__ == '__main__':
    articleUsers = []
    for i in range(5):
        articleUsers.append(ArticleStruct())

    currentArticleUser = articleUsers[0]
    currentArticleUser.articlePickedList = data
    currentArticleUser.articleClickedList = clickState

    result = currentArticleUser.calculateSim(testArticle[0])
    tempSum = 0
    for i in range(len(result)):
        tempSum += result[i]

    print tempSum
    print result
