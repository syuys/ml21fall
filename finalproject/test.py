# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 22:16:26 2022

@author: Eric
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% main
dataSetPath = "dataset"

trainIDSet = pd.read_csv(os.path.join(dataSetPath, "Train_IDs.csv"))
testIDSet = pd.read_csv(os.path.join(dataSetPath, "Test_IDs.csv"))
statusSet = pd.read_csv(os.path.join(dataSetPath, "status.csv"))

# check if ID repeat
for testID in testIDSet["Customer ID"].values:
    if testID in trainIDSet["Customer ID"].values:
        print("Repeat !!")

# check occurrence frequency of each churn category
categories, counts = np.unique(statusSet["Churn Category"].values, return_counts=True)
plt.pie(counts, labels=categories, autopct='%1.1f%%')
plt.title("The proportion of each category")
plt.show()