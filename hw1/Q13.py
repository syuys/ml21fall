#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 05:37:08 2021

@author: YI SIANG SYU

Description:
    Q13.
    There are four main parts in this .py file. First is to import necessary packages. 
    Second is to define function I used in PLA algorithm. Third is to load data and do
    some preprocessing like adding x0 to every xn. Fourth is the key part for doing
    PLA loop (each experiment is done with different random seed).
    For this question, there is no scaling factor.
"""


# %% First: import necessary packages
from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import numpy as np
from datetime import datetime


# %% Second: define function
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


# %% Third: preprocess
# load data
dataSet = np.genfromtxt("hw1_train.dat")

# insert x0, extract x vector set and y set
x0 = 1
dataSet = np.insert(dataSet, 0, x0, axis=1)
xSet = dataSet[:, :-1]
ySet = dataSet[:, -1]

# set other related parameters
cumuCorrectThold = 5*dataSet.shape[0]
repeatTestTimes = 1000
wSet = np.empty((repeatTestTimes, xSet.shape[1]))  # container for storing wPLA


# %% Fourth: main - do experiment
# get current time, used for seed setting.
currentTime = int(datetime.now().strftime("%H:%M:%S").replace(":", ""))
print("Start to do PLA!")
print("\nMonitor progress:")
for testIdx in range(repeatTestTimes):  # Repeat experiment for 1000 times
    # initialization
    t = 0
    cumuCorrect = 0
    w = np.zeros(xSet.shape[1])  # initialize w
    # ensure each experiment is done with different seed.
    np.random.seed(int(currentTime*1e4)+testIdx)
    # do PLA
    while(True):
        # randomly picks an example in every iteration (with replacement)
        idx = np.random.randint(0, xSet.shape[0])
        if getH(w, xSet[idx]) != ySet[idx]:
            cumuCorrect = 0
            # updates w if and only if w is incorrect on the example.
            w = w + ySet[idx]*xSet[idx]
        else:
            cumuCorrect += 1
            if cumuCorrect >= cumuCorrectThold:
                # stop updating and return w as wPLA if w is correct consecutively
                # after checking 5N randomly-picked examples.
                break
        t += 1
    wSet[testIdx] = w
    print("{}, ".format(testIdx), end="")
print("\n\nDone!")
# show the average squared length of wPLA
print("Average squared length of wPLA:", (wSet**2).sum(axis=1).mean())