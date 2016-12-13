import numpy as np
import os
from conf import *
from matplotlib.pylab import *
from operator import itemgetter

if __name__ == '__main__':

    filename ="./LastFMResults/"+"LastFM_50_shuffled_Clustering_ALL_Diagnol_1__09_15_00_40.csv"

    CoLinReward = {}
    GOBLinReward = {}
    RandomReward = {}
    Uniform_LinUCBReward = {}
    LinUCBReward = {}
    Temp1 = {}
    Temp2 = {}

    CoLinRatio = {}
    Uniform_LinUCBRatio = {}
    LinUCBRatio = {}
    GOBLinRatio = {}

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
            RandomReward[i], LinUCBReward[i], CoLinReward[i], GOBLinReward[i] = [float(x) for x in words[2].split(';')]
            # RandomReward[i], Uniform_LinUCBReward[i], LinUCBReward[i], CoLinReward[i], GOBLinReward[i] = [float(x) for x in words[2].split(';')]

            # tim[i] = int(words[1])
            tim[i] = i

    print len(tim), len(GOBLinReward)
    plt.plot(tim.values(), CoLinReward.values(), label='CoLin')
    plt.plot(tim.values(), RandomReward.values(), label='RandomReward')

    plt.plot(tim.values(), LinUCBReward.values(), label='N_LinUCB')
    plt.plot(tim.values(), GOBLinReward.values(), label='GOBLin')
    # plt.plot(GOBtim.values(), GOBLinCTRRatio.values(),  label = 'GOB.Lin')
    # plt.plot(tim.values(), LinUCBCTRRatio.values(), label = 'LinUCB')
    plt.xlabel('time')
    plt.ylabel('Reward')
    plt.legend(loc='upper left')
    plt.title('LastFM_UserNum' + str(userNum))
    plt.show()
