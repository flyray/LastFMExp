import numpy as np

M = [[1, 2, 3, 4],
     [2, 3, 4, 5],
     [3, 4, 5, 6]]

if __name__ == '__main__':
    articleList = []

    # print 'articleList[0]', articleList[0]
    articleList.append(1)
    articleList.append(2)
    for i in range(len(articleList)):
        print 'i: ', i
        a = M[0][i]
        b = M[i][i]
        print b

    print 'END!'

    if 1000 % 10 == 0:
        print "ok"