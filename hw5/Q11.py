# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 07:01:01 2021

@author: Eric
"""

import numpy as np
import pandas as pd
from libsvm.svmutil import *


# %% Q11 - Main
# read data
trainY, trainX = svm_read_problem("satimage.scale.txt", return_scipy=True)

# re-label
trainY[trainY != 5] = -1
trainY[trainY == 5] = 1

# training
m = svm_train(trainY, trainX, "-s 0 -t 0 -c 10")

# retreive support vectors
sv = m.get_SV()
sv = pd.DataFrame(sv)
sv = sv.sort_index(axis=1)

# replace nan values with zeros
sv = sv.fillna(0)

# retreive support vector coefficients
svCoef = m.get_sv_coef()
svCoef = np.array(svCoef)

# calculate weight and its norm
w = svCoef.T @ sv
w = w.values.ravel()
wNorm = np.sum(w**2) ** (0.5)
print("wNorm:", wNorm)  # 4.645