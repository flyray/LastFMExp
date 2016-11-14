import numpy as np
import os
from conf import *
from matplotlib.pylab import *
from operator import itemgetter
import sys

if __name__ == '__main__':
    Yahoo_save_address = str(sys.argv[1])

    filenames = [x for x in os.listdir(Yahoo_save_address) if '.csv' in x]

    algName = {}
    algNum = int(sys.argv[2])
    UserNum = int(sys.argv[3])
    for a in range(algNum):
        algName[a] = sys.argv[4 + a]
    for a in range(algNum):

        AlgCTR = {}
        RandomCTR = {}
        AlgCTRRatio = {}

        tim = {}

        filenames.sort()
        i = -1
        for x in filenames:
            filename = os.path.join(Yahoo_save_address, x)
            if str(algName[a]) in x:
                with open(filename, 'r') as f:
                    print str(filename)
                    for line in f:
                        i = i + 1
                        words = line.split(',')
                        if words[0].strip() != 'data':
                            continue
                        RandomCTR[i], AlgCTR[i] = [float(x) for x in words[2].split(';')]
                        AlgCTRRatio[i] = AlgCTR[i] / RandomCTR[i]

                        tim[i] = i

        plt.plot(tim.values(), AlgCTRRatio.values(), label=str(algName[a]))

    plt.xlabel('time')
    plt.ylabel('CTR-Ratio')
    plt.legend(loc='lower right')
    plt.title('UserNum =' + str(UserNum))
    plt.show()

    '''      
    plt.plot(tim.values(), ucbCTRRatio.values(), label = 'Restart_ucbCTR Ratio')
    plt.plot(tim.values(), greedyCTRRatio.values(),  label = 'greedyCTR Ratio')
    plt.legend()
    plt.show()
    '''
