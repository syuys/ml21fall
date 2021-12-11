# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 23:10:09 2021

@author: Eric
"""

import numpy as np
import itertools
from liblinear.liblinearutil import *


def polynomialTransform(x, orderQ):
    dim = x.shape[1]
    # add x0 = 1
    z = np.insert(x, 0, 1, axis=1)
    # add higher order
    for num in range(2, orderQ+1):
        combSet = np.array(list(itertools.combinations_with_replacement(np.arange(dim), num)))
        for comb in combSet:
            # multiplyTerm = x[:, comb[0]] * x[:, comb[1]]
            multiplyTerm = np.prod(x[:, comb], axis=1)
            z = np.concatenate((z, multiplyTerm[:, None]), axis=1)
    return z


# parameters
trainDataPath = "hw4_train.txt"
testDataPath = "hw4_test.txt"
Q = 3
log10LambdaSet = [-4, -2, 0, 2, 4]
Vfold = 5


# read data
trainData = np.genfromtxt(trainDataPath)
testData = np.genfromtxt(testDataPath)


# full order polynomial transform and split data
trainData = np.concatenate((polynomialTransform(trainData[:, :-1], orderQ=Q), trainData[:, -1][:, None]), axis=1)
testData = np.concatenate((polynomialTransform(testData[:, :-1], orderQ=Q), testData[:, -1][:, None]), axis=1)


# model training and evaluating Ecv
trainData = trainData.reshape(Vfold, trainData.shape[0]//Vfold, -1)
modelCandidates = []
DvalIdxSet = [*range(Vfold)]
EcvSet = []
for log10Lambda in log10LambdaSet:
    print("\nlog10Lambda:", log10Lambda)
    # do cross validation
    EvalSet = []
    for DvalIdx in DvalIdxSet:
        # split data
        Dtrain = trainData[np.delete(DvalIdxSet, DvalIdx), :, :].reshape(-1, trainData.shape[-1])
        Dval = trainData[DvalIdx, :, :].reshape(-1, trainData.shape[-1])
        # train model
        actualLambda = 10**(log10Lambda)
        C = 1 / (2*actualLambda)
        prob = problem(Dtrain[:, -1], Dtrain[:, :-1])
        param = parameter('-s 0 -c {} -e 0.000001'.format(C))
        model = train(prob, param)
        # evaluate Eval
        p_label, p_acc, p_val = predict(Dval[:, -1], Dval[:, :-1], model)
        Eval = 100-p_acc[0]
        EvalSet.append(Eval)
    EcvSet.append(np.mean(EvalSet)/100)
print("\nEcvSet:", EcvSet)