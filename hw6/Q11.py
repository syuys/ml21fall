# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 19:55:42 2022

@author: Eric
"""

import numpy as np


# get predicted value from h
def getH(x, s, i, theta):
    x = x.reshape(-1, x.shape[-1])  # reshape to 2D
    h = s * np.sign(x[i, :] - theta)  # ignore the case of sign(0)
    return h


# calculate weighted 0/1 error
def getWeightedError(predictedValue, trueValue, u):
    weightedError = (predictedValue != trueValue) @ u / (np.ones(u.size) @ u) 
    return  weightedError


# decision stump algorithm - find best (s, i, theta)
def getG(data, u, thetaCandidates):
    # setting
    x = data[:, :, 0]  # [10, 1000]
    y = data[:, :, -1]  # [10, 1000]
    s = np.array([-1, 1])
    targetFeatures = np.arange(x.shape[0])
    error = np.empty((s.size, targetFeatures.size, thetaCandidates.shape[1]))  # [2, 10, 1000]
    
    # calculate error on theta = -inf
    predictedValues = getH(x=x,
                           s=s.reshape(-1, 1, 1),
                           i=targetFeatures,
                           theta=-np.inf
                           )  # [2, 10, 1000]
    error[:, :, 0] = getWeightedError(predictedValues, y, u)  # [2, 10]
    
    # iteratively get error on remaining theta (all midpoints)
    for thetaIdx in range(1, thetaCandidates.shape[1]):
        error[0, :, thetaIdx] = error[0, :, thetaIdx-1] - y[:, thetaIdx-1] * u[thetaIdx-1]  # for s = -1
        error[1, :, thetaIdx] = error[1, :, thetaIdx-1] + y[:, thetaIdx-1] * u[thetaIdx-1]  # for s = +1
        
    # get the best (s, i, theta) combination
    minError = error.min()
    bestSIdx, bestIIdx, bestThetaIdx = np.where(error == minError)
    bestS, bestI, bestTheta = s[bestSIdx[0]], targetFeatures[bestIIdx[0]], thetaCandidates[bestIIdx[0]][bestThetaIdx[0]]
    
    return bestS, bestI, bestTheta, minError, error


def getAggregatedG():
    return None


# parameters
trainDataPath = "hw6_train.txt"
testDataPath = "hw6_test.txt"


# read data
trainData = np.genfromtxt(trainDataPath)
testData = np.genfromtxt(testDataPath)


# check if all the ith features are unique
for col in range(trainData.shape[-1]-1):
    print("col_{}: {}".format(col, len(np.unique(trainData[:, col]))))


# sort training data
sampleNum = trainData.shape[0]
featureNum = trainData.shape[-1]-1
trainDataSorted = np.empty((featureNum, sampleNum, 2))  # 2 means the ith feature and corresponding label
for i in range(featureNum):
    ithData = trainData[:, [i, -1]]
    trainDataSorted[i] = ithData[ithData[:, 0].argsort()]


# get threshold candidates of each feature
thetaCandidates = trainDataSorted[:, :, 0]
thetaCandidates = (thetaCandidates[:, :-1] + thetaCandidates[:, 1:]) / 2
thetaCandidates = np.insert(thetaCandidates, 0, -np.inf, axis=1)  # [10, 1000]

# main
u = np.ones(trainDataSorted.shape[1]) / trainDataSorted.shape[1]  # initialize u
s, i , theta, eIn, eInSet = getG(trainDataSorted, u, thetaCandidates)










