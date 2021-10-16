#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 07:19:04 2021

@author: YI SIANG SYU
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import matplotlib.pyplot as plt
plt.close("all")
import numpy as np
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% define function
def getH(w, x):
    # inner product
    innerProduct = np.matmul(w, x.T)
    # get sign
    if innerProduct > 0:
        h = 1
    elif innerProduct < 0:
        h = -1
    else:  # %% case of innerProduct = 0
        h = -1
    return h    


# %% 

# load data
dataSet = np.genfromtxt("hw1_train.dat")

# insert x0, extract x vector set and y set
x0 = 1
dataSet = np.insert(dataSet, 0, x0, axis=1)
xSet = dataSet[:, :-1]
xSet = xSet*2  # scaling by factor of 2 for question 14
ySet = dataSet[:, -1]

# set other related parameters
cumuCorrectThold = 5*dataSet.shape[0]
repeatTestTimes = 1000
wSet = np.empty((repeatTestTimes, xSet.shape[1]))


# %% preprocess



# %% main - do experiment
for testIdx in range(1000):
    # initialization
    t = 0
    cumuCorrect = 0
    w = np.zeros(xSet.shape[1])  # initial w
    # do PLA
    while(True):
        idx = np.random.randint(0, xSet.shape[0])
        if getH(w, xSet[idx]) != ySet[idx]:
            cumuCorrect = 0
            w = w + ySet[idx]*xSet[idx]
        else:
            cumuCorrect += 1
            if cumuCorrect >= cumuCorrectThold:
                break
        t += 1
    wSet[testIdx] = w
    print("{}, ".format(testIdx), end="")
print("\n\n")
print("Average squared length of wPLA:", (wSet**2).sum(axis=1).mean())