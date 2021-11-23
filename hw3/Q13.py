#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 20:56:23 2021

@author: md703
"""

import numpy as np


def polyQ(x, degree):
    # add x0 = 1 first
    z = np.insert(x, 0, 1, axis=1)
    # do transform for order-degree
    for q in range(2, degree+1):
        z = np.concatenate((z, x**q), axis=1)
    return z


# parameters
Q = 8
trainDataPath = "hw3_train.dat"
testDataPath = "hw3_test.dat"

# read data
trainData = np.genfromtxt(trainDataPath)
testData = np.genfromtxt(testDataPath)

# homogeneous order-Q polynomial transform
trainData = np.concatenate((polyQ(trainData[:, :-1], degree=Q), trainData[:, -1][:, None]), axis=1)
testData = np.concatenate((polyQ(testData[:, :-1], degree=Q), testData[:, -1][:, None]), axis=1)

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