# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 08:52:33 2021

@author: Eric
"""

from concurrent.futures import ThreadPoolExecutor
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

# param set
gammaSet = [0.1, 1, 10, 100, 1000]  # gamma candidates
bestGamma = []  # container for storing best gamma in each experiment
T = 1000  # total times for repeating validation procedure

# container for storing best gamma in each experiment


def validate(idx):
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
        p_label, p_acc, p_val = svm_predict(DvalY, DvalX, m, options="-q")
        EvalSet.append(100 - p_acc[0])
    
    # storing the best gamma in the current experiment
    bestGamma.append(gammaSet[np.argmin(EvalSet)])

np.random.seed(5)  # compare with single-thread
with ThreadPoolExecutor() as executor:
    for expIdx in range(T):
        executor.submit(validate, expIdx)
    

# analyze the selected number of each gamma
unique, counts = np.unique(bestGamma, return_counts=True)
result = dict(zip(unique, counts))
print()
print("result:", result)