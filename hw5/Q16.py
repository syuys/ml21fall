# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 09:49:32 2021

@author: Eric
"""

import numpy as np
from libsvm.svmutil import *


# %% Q16 - Main
# read data
trainY, trainX = svm_read_problem("satimage.scale.txt")

# re-label
trainX = np.array(trainX)
trainY = np.array(trainY)
trainY[trainY != 1] = -1
trainY[trainY == 1] = 1

# set gamma candidates
gammaSet = [0.1, 1, 10, 100, 1000]

# container for storing best gamma in each experiment
bestGamma = []

np.random.seed(5)  # compare with multi-thread
for expIdx in range(1000):
    print("============ exp: {}/1000 ============".format(expIdx))
    
    # randomly samples 200 examples for creating Dtrain and Dval
    idxSet = np.random.choice(np.arange(trainX.size), size=200, replace=False)
    DtrainX = list(np.delete(trainX, idxSet, 0))
    DtrainY = list(np.delete(trainY, idxSet))
    DvalX = list(trainX[idxSet])
    DvalY = list(trainY[idxSet])
    
    # container for storing Eval corresponding to each gamma
    EvalSet = []
    
    # iterate over each gamma parameter
    for gamma in gammaSet:
        # training
        m = svm_train(DtrainY, DtrainX, "-s 0 -t 2 -g {} -c 0.1".format(gamma))
        
        # evaluating Eval
        p_label, p_acc, p_val = svm_predict(DvalY, DvalX, m)
        EvalSet.append(100 - p_acc[0])
    
    # storing the best gamma in the current experiment
    bestGamma.append(gammaSet[np.argmin(EvalSet)])

# analyze the selected number of each gamma
unique, counts = np.unique(bestGamma, return_counts=True)
result = dict(zip(unique, counts))
print("result:", result)