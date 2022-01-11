# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 09:09:43 2021

@author: Eric
"""

import numpy as np
import copy
from libsvm.svmutil import *


# %% Q12 - Main
# read data
trainY, trainX = svm_read_problem("satimage.scale.txt", return_scipy=True)

# iterate over each classifier
for target in [2, 3, 4, 5, 6]:
    # re-label
    currentY = copy.deepcopy(trainY)
    currentY[currentY != target] = -1
    currentY[currentY == target] = 1
    
    # training
    m = svm_train(currentY, trainX, "-s 0 -t 1 -d 3 -g 1 -r 1 -c 10")
    
    # show acc and Ein information
    print("Target:", target)    
    p_label, p_acc, p_val = svm_predict(currentY, trainX, m)
    Ein = 100 - p_acc[0]
    print("Ein: {}%".format(np.round(Ein, 2)))
    print()