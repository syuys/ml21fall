# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:12:59 2022

@author: Eric
"""

import time
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# %% load data
dataSetPath = "dataset"
trainData = pd.read_csv(os.path.join(dataSetPath, "merge_after_preprocessing.csv"), index_col=[0])


# %% process raw feature
y_train = trainData['Churn Category'].values
X_train = trainData.drop('Customer ID', axis=1)
X_train = X_train.drop('Churn Category', axis=1)

# standard normalized
ss = StandardScaler().fit(X_train)
X_train_std = ss.transform(X_train)
# normalized to (0, 1)
mms = MinMaxScaler(feature_range=(0, 1)).fit(X_train_std)
X_train_std = mms.transform(X_train_std)