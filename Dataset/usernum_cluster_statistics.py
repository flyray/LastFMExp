import sys
import matplotlib.pyplot as plt
import numpy as np

labelfile = sys.argv[1]

# read cluster label
label = []
for i in range(200):
	label.append((0, i))
with open(labelfile,'r') as fin:
	for line in fin:		
	    label[int(line)] = (label[int(line)][0]+1, label[int(line)][1])

label.sort()
label.reverse()
print [x[1] for x in label]
plt.plot(range(1, 201), [x[0] for x in label])
plt.show()