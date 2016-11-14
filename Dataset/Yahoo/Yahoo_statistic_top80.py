import sys

usernum = [0 for i in range(160)]
with open(sys.argv[1], 'r') as f:
    # reading file line ie observations running one at a time
    for line in f:
        line = line.split("|")
        userID = int(line[1])
        usernum[userID] += 1

with open(sys.argv[1] + '.statistic', 'w') as fout:
    fout.write('\n'.join(str(x) for x in usernum))
