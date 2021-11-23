#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 21:00:35 2021

@author: md703
"""

import numpy as np
import itertools


def getFullOrder2(x):
    dim = x.shape[1]
    # add x0 = 1
    z = np.insert(x, 0, 1, axis=1)
    # add x1x2, x2x3, ...
    combSet = np.array(list(itertools.combinations(np.arange(dim), 2)))
    for comb in combSet:
        multiplyTerm = x[:, comb[0]] * x[:, comb[1]]
        z = np.concatenate((z, multiplyTerm[:, None]), axis=1)
    # add x1^2, x2^2, ...
    z = np.concatenate((z, x**2), axis=1)
    return z


# parameters
trainDataPath = "hw3_train.dat"
testDataPath = "hw3_test.dat"

# read data
trainData = np.genfromtxt(trainDataPath)
testData = np.genfromtxt(testDataPath)

# homogeneous order-Q polynomial transform
trainData = np.concatenate((getFullOrder2(trainData[:, :-1]), trainData[:, -1][:, None]), axis=1)
testData = np.concatenate((getFullOrder2(testData[:, :-1]), testData[:, -1][:, None]), axis=1)

# do linear regression
dagger = np.linalg.pinv(trainData[:, :-1])
wlin = np.matmul(dagger, trainData[:, -1])

# calculate Ein w.r.t wlin
trainPred = np.matmul(trainData[:, :-1], wlin)
trainPred[trainPred>=0] = 1
trainPred[trainPred<0] = -1
Ein = np.mean(trainPred != trainData[:, -1])
    
# calculate Eout w.r.t wlin
testPred = np.matmul(testData[:, :-1], wlin)
testPred[testPred>=0] = 1
testPred[testPred<0] = -1
Eout = np.mean(testPred != testData[:, -1])

# save final result
result = abs(Ein-Eout)