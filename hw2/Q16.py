#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 00:00:58 2021

@author: md703
"""

import numpy as np

def generateSamples(N):
    samples = np.empty((N, 4))  # 4 --> x0, x1, x2, y
    # y
    samples[:, 3] = np.random.randint(0, 2, N)
    samples[:, 3] = samples[:, -1]*2 - 1
    # x0
    samples[:, 0] = 1
    # x1, x2
    samples[samples[:, 3]==1, 1:3] = +\
        np.random.multivariate_normal(mean=[2, 3], 
                                      cov=[[0.6, 0], [0, 0.6]], 
                                      size=sum(samples[:, 3]==1))
    samples[samples[:, 3]==-1, 1:3] = +\
        np.random.multivariate_normal(mean=[0, 4], 
                                      cov=[[0.4, 0], [0, 0.4]], 
                                      size=sum(samples[:, 3]==-1))
    return samples


def generateOutliers(N):
    outLiers = np.empty((N, 4))  # 4 --> x0, x1, x2, y
    # y
    outLiers[:, 3] = 1
    # x0
    outLiers[:, 0] = 1
    # x1, x2
    outLiers[:, 1:3] = +\
        np.random.multivariate_normal(mean=[6, 0], 
                                      cov=[[0.3, 0], [0, 0.1]], 
                                      size=sum(outLiers[:, 3]==1))
    return outLiers


def sigmoid(s):
    return 1/(1+np.exp(-s))


if __name__ == '__main__':
    # parameters
    trainN = 200
    outliersN = 20
    testN = 5000
    eta = 0.1
    T = 500
    repeatTimes = 100
    
    # repeat process
    resultSet = np.empty((repeatTimes, 2))
    for idx in range(repeatTimes):
        # generate data with a specific seed
        np.random.seed(idx)
        train = generateSamples(trainN)
        outliers = generateOutliers(outliersN)  # generate outliers
        train = np.concatenate((train, outliers), axis=0)  # add outliers
        test = generateSamples(testN)
        
        # linear regression
        dagger = np.linalg.pinv(train[:, :3])
        wlin = np.matmul(dagger, train[:, 3])
        testPred = np.matmul(test[:, :3], wlin)
        testPred[testPred>=0] = 1
        testPred[testPred<0] = -1
        EoutLinregress = np.mean(testPred != test[:, 3])
        
        # logistic regression
        w = np.zeros(train.shape[1]-1)
        for _ in range(T):
            gradient = np.mean(sigmoid(-train[:, -1] * np.matmul(w, train[:, :3].T)) * 
                               (-train[:, -1].reshape(-1, 1)*train[:, :3]).T, axis=1)
            w = w - eta*gradient
        testPred = sigmoid(np.matmul(w, test[:, :3].T))
        testPred[testPred>=0.5] = 1
        testPred[testPred<0.5] = -1
        EoutLogregress = np.mean(testPred != test[:, 3])
        
        # save the result of this process
        resultSet[idx] = [EoutLinregress, EoutLogregress]

# final result
resultSetMean = resultSet.mean(axis=0)