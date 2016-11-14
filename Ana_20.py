import numpy as np
import os
from conf import *
from matplotlib.pylab import *
from operator import itemgetter

if __name__ == '__main__':

    filenames = [x for x in os.listdir(LastFM_save_address) if '.csv' in x]

    CoLinReward = {}
    GOBLinReward = {}
    RandomReward = {}
    LinUCBReward = {}
    Temp1 = {}
    Temp2 = {}

    CoLinRatio = {}
    LinUCBRatio = {}
    GOBLinRatio = {}

    tim = {}
    GOBtim = {}
    # i = -1
    for x in filenames:
        filename = os.path.join(LastFM_save_address, x)
        if 'LastFM_100_shu' in x:
            i = -1
            with open(filename, 'r') as f:
                print str(filename)
                for line in f:

                    i = i + 1
                    words = line.split(',')
                    if words[0].strip() != 'data':
                        continue
                    RandomReward[i], CoLinReward[i], LinUCBReward[i], GOBLinReward[i] = [float(x) for x in
                                                                                         words[2].split(';')]
                    '''
                    CoLinRatio[i] = CoLinReward[i]/RandomReward[i]
                    LinUCBRatio[i] = LinUCBReward[i]/RandomReward[i]
                    GOBLinRatio[i] = GOBLinReward[i]/RandomReward[i]
                    '''
                    # tim[i] = int(words[1])
                    tim[i] = i

    # print 'len(tim)', len(tim), 'len', len(ucbCTRRatio)
    # print ucbCTRRatio
    print len(tim), len(GOBLinReward)

    plt.plot(tim.values(), CoLinReward.values(), label='CoLin')
    plt.plot(tim.values(), LinUCBReward.values(), label='LinUCB')
    plt.plot(tim.values(), GOBLinReward.values(), label='GOBLin')
    # plt.plot(GOBtim.values(), GOBLinCTRRatio.values(),  label = 'GOB.Lin')
    # plt.plot(tim.values(), LinUCBCTRRatio.values(), label = 'LinUCB')
    plt.xlabel('time')
    plt.ylabel('CTR-Ratio')
    plt.legend(loc='lower right')
    plt.title('UserNum = 100')
    plt.show()

    '''      
    plt.plot(tim.values(), ucbCTRRatio.values(), label = 'Restart_ucbCTR Ratio')
    plt.plot(tim.values(), greedyCTRRatio.values(),  label = 'greedyCTR Ratio')
    plt.legend()
    plt.show()
    '''
