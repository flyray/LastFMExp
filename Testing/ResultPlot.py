__author__ = 'Dianlei Zhang'
from matplotlib.pylab import *

filename ="../LastFMResults/"+"LastFM__shuffled_Clustering_LinUCB_Diagnol_Origin_processed_events_shuffled.dat_IniW2000_12_22_15_48_47.csv"
RandomReward = {}
LinUCBReward = {}
TimeStamp = {}
tim = {}
GOBtim = {}
userNum = 100
i = -1
with open(filename, 'r') as f:
    print str(filename)
    for line in f:
        i += 1
        words = line.split(',')
        if words[0].strip() != 'data':
            continue
        RandomReward[i], TimeStamp[i], LinUCBReward[i] = [float(x) for x in words[2].split(';')]
        tim[i] = i

print len(tim), len(LinUCBReward)
plt.plot(tim.values(), RandomReward.values(), label='RandomReward')
plt.plot(tim.values(), LinUCBReward.values(), label='LinUCB')
plt.xlabel('time')
plt.ylabel('Reward')
plt.legend(loc='upper left')
plt.title('LastFM_UserNum' + str(userNum))
plt.show()