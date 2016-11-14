import sys
from sets import Set
import random

usernum = [(0, i) for i in range(160)]

yahooData_address = sys.argv[1]
dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
for dataDay in dataDays:
    fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay + '.userID.statistic'
    with open(fileName, 'r') as f:
        for i, line in enumerate(f):
            usernum[i] = (usernum[i][0] + int(line), i)

usernum.sort()
usernum.reverse()
a = [x[1] for x in usernum]
a = a[:80]

with open('YahooKMeansModel/10kmeans_model160.dat', 'r') as fin:
    kmeans = fin.read().split('\n')
with open('YahooKMeansModel/10kmeans_model160.dat.80.max', 'w') as fout:
    for x in a:
        fout.write(kmeans[x] + '\n')

with open(sys.argv[2], 'r') as fin:
    cut_rd = [int(x) for x in fin.readline().strip('[]\n').split(',')]
    cut_mx = [int(x) for x in fin.readline().strip('[]\n').split(',')]

dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
for dataDay in dataDays:
    fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay + '.userID'

    fout_Part1 = open(fileName + '.max.part1', 'w')
    fout_Part2 = open(fileName + '.max.part2', 'w')

    with open(fileName, 'r') as fin:
        for line in fin:
            line = line.split("|")
            userID = int(line[1])
            if userID in a:
                line[1] = str(a.index(userID))
                if cut_mx[a.index(userID)] == 1:
                    fout_Part1.write('|'.join(line))
                else:
                    fout_Part2.write('|'.join(line))
    fout_Part1.close()
    fout_Part2.close()
