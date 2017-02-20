
a = [1, 2, 3, 4]

if 1 in a:
    print "ok"

if 5 in a:
    print '5 in a'

print a.index(2)

totalList = []
i = 0
for i in range(100):
    totalList.append([])

totalList[0].append([1, 2, 4])
totalList[3].append(2)

print 'totalList[3]:', totalList[3]
print 'totalList[99]:', totalList[0]
print 'totalList length:', len(totalList)
