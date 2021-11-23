#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 21:31:14 2021

@author: md703
"""

import numpy as np


def getTransform(x, leftDim):
    # transform to a lower dimension
    x = x[:, :leftDim]
    # add x0 = 1
    z = np.insert(x, 0, 1, axis=1)    
    return z


# parameters
trainDataPath = "hw3_train.dat"
testDataPath = "hw3_test.dat"

# read data
rawTrainData = np.genfromtxt(trainDataPath)
rawTestData = np.genfromtxt(testDataPath)

# main - evaluate each transform function
resultSet = []
for i in range(1, 11):
    # transform both training and testing dataset to a lower dimensional space
    trainData = np.concatenate((getTransform(rawTrainData[:, :-1], leftDim=i), rawTrainData[:, -1][:, None]), axis=1)
    testData = np.concatenate((getTransform(rawTestData[:, :-1], leftDim=i), rawTestData[:, -1][:, None]), axis=1)
    
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
    
    # save result
    resultSet.append(abs(Ein-Eout))

# save final best idx
bestIdx = np.argmin(resultSet)