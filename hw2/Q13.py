#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 10:09:13 2021

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

EinSet = []
for _ in range(100):
    train = generateSamples(200)
    dagger = np.linalg.pinv(train[:, :3])
    wlin = np.matmul(dagger, train[:, 3])
    # append squared-error for this wlin
    EinSet.append(np.square(np.matmul(train[:, :3], wlin) - train[:, 3]).mean())

EinSetMean = np.array(EinSet).mean()