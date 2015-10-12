import sys
from sets import Set
import random

usernum = [(0, i) for i in range(160)]

yahooData_address = sys.argv[1].split('/')[0]
dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
for dataDay in dataDays:
		fileName = yahooData_address + "/ydata-fp-td-clicks-v1_0.200905" + dataDay	+'.userID.statistic'
		with open(fileName, 'r') as f:
			for i, line in enumerate(f):
				usernum[i] = (usernum[i][0]+int(line), i)

usernum.sort()
usernum.reverse()
a = [x[1] for x in usernum]
a = a[:80]

with open(sys.argv[2],'r') as fin:
	cut_rd = [int(x) for x in fin.readline().strip('[]\n').split(',')]
	cut_mx = [int(x) for x in fin.readline().strip('[]\n').split(',')]
	
fout_Part1 = open(sys.argv[1]+'.max.part1','w')
fout_Part2 = open(sys.argv[1]+'.max.part2','w')

with open (sys.argv[1], 'r') as fin:
	for line in fin:	
		line = line.split("|")
		userID = int(line[1])
		if userID in a:
			if cut_mx[a.index(userID)] == 1:
				fout_Part1.write('|'.join(line))
			else:
				fout_Part2.write('|'.join(line))
fout_Part1.close()
fout_Part2.close()


