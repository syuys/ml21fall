# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 09:28:39 2021

@author: Eric
"""

import numpy as np
import copy
from libsvm.svmutil import *


# %% Q14 - Main
# read data
trainY, trainX = svm_read_problem("satimage.scale.txt", return_scipy=True)
testY, testX = svm_read_problem("satimage.scale.t.txt", return_scipy=True)

# re-label
trainY[trainY != 1] = -1
trainY[trainY == 1] = 1
testY[testY != 1] = -1
testY[testY == 1] = 1

# iterate over each C parameter
for C in [0.01, 0.1, 1, 10, 100]:
    # training
    m = svm_train(trainY, trainX, "-s 0 -t 2 -g 10 -c {}".format(C))
    
    # show the current C and corresponding Eout
    print("C:", C)    
    p_label, p_acc, p_val = svm_predict(testY, testX, m)
    Eout = 100 - p_acc[0]
    print("Eout: {}%".format(np.round(Eout, 2)))
    print()