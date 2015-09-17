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
    #i = -1 
    for x in filenames:    
        filename = os.path.join(LastFM_save_address, x)
        if 'LastFM_100_shuffled' in x:
            i = -1 
            with open(filename, 'r') as f:
                print str(filename)      
                for line in f:
                   
                    i = i + 1
                    words = line.split(',')
                    if words[0].strip() != 'data':
                        continue
                    RandomReward[i], Uniform_LinUCBReward[i], LinUCBReward[i], CoLinReward[i],GOBLinReward[i]= [float(x) for x in words[2].split(';')]

                    #tim[i] = int(words[1])
                    tim[i] = i

    print len(tim), len(GOBLinReward)
    plt.plot(tim.values(), CoLinReward.values(), label = 'CoLin')
    plt.plot(tim.values(), Uniform_LinUCBReward.values(), label = 'Uniform_LinUCB')
    
    plt.plot(tim.values(), LinUCBReward.values(), label = 'N_LinUCB')
    plt.plot(tim.values(), GOBLinReward.values(), label = 'GOBLin')
    #plt.plot(GOBtim.values(), GOBLinCTRRatio.values(),  label = 'GOB.Lin')
    #plt.plot(tim.values(), LinUCBCTRRatio.values(), label = 'LinUCB')
    plt.xlabel('time')
    plt.ylabel('Reward')
    plt.legend(loc = 'lower right')
    plt.title('LastFM_UserNum'+str(userNum))
    plt.show()


   