#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 11:16:01 2021

@author: md703
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

def generateSamples(N):
    samples = np.empty((N, 4))  # 4 --> x0, x1, x2, y
    # y
    samples[:, 3] = np.random.randint(0, 2, N)
    samples[:, 3] = samples[:, -1]*2 - 1
    # x0
    samples[:, 0] = 1
    # x1, x2
    samples[samples[:, 3]==1, 1:3] = np.random.multivariate_normal(mean=[2, 3], cov=[[0.6, 0], [0, 0.6]], size=sum(samples[:, 3]==1))
    samples[samples[:, 3]==-1, 1:3] = np.random.multivariate_normal(mean=[0, 4], cov=[[0.4, 0], [0, 0.4]], size=sum(samples[:, 3]==-1))
    return samples

differenceSet = []
for _ in range(100):
    train = generateSamples(200)
    test = generateSamples(5000)
    dagger = np.linalg.pinv(train[:, :3])
    wlin = np.matmul(dagger, train[:, 3])
    # calculate Ein for this wlin
    trainPred = np.matmul(train[:, :3], wlin)
    trainPred[trainPred>=0] = 1
    trainPred[trainPred<0] = -1
    Ein = np.mean(trainPred != train[:, 3])
    # calculate Eout for this wlin
    testPred = np.matmul(test[:, :3], wlin)
    testPred[testPred>=0] = 1
    testPred[testPred<0] = -1
    Eout = np.mean(testPred != test[:, 3])
    # append difference
    differenceSet.append(abs(Ein-Eout))
    

differenceSetMean = np.array(differenceSet).mean()