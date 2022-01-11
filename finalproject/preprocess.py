# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 01:36:40 2022

@author: Eric
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %% function
def replaceNanWithMost(col):
    freq = merge.groupby(col).size() 

    names = [name for name, _ in freq.items()]
    counts = [count for _, count in freq.items()]

    # Replace NaN with the most frequent label
    merge_copy[col] = merge_copy[col].fillna(names[counts.index(max(counts))])


def replaceNanWithAvg(col):
    if 'Count' in col:
        merge_copy[col] = merge_copy[col].fillna(0)
    else:
        merge_copy[col] = merge_copy[col].fillna(merge_copy[col].mean())


# %% load data
dataSetPath = "dataset"

merge = pd.read_csv(os.path.join(dataSetPath, "merge.csv"), index_col=[0])
merge_cols = merge.columns
merge_copy = merge.copy(deep=True)


# %% Replace NaN with the most frequent label
colReplaceMost = ['Churn Category', 'Satisfaction Score', 
       'Gender', 'Under 30', 'Senior Citizen', 'Married', 'Dependents',
       'Number of Dependents', 'Country', 'State', 'City', 'Quarter',
       'Referred a Friend', 'Number of Referrals', 'Offer',
       'Phone Service', 'Multiple Lines', 'Internet Service', 'Internet Type',
       'Online Security', 'Online Backup', 'Device Protection Plan',
       'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
       'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing',
       'Payment Method']

for col in colReplaceMost:
    replaceNanWithMost(col)

    # Encode target labels with value
    le = LabelEncoder()
    merge_copy[col] = le.fit_transform(merge_copy[col])

    if col == 'Churn Category':
        encoder_map = dict(zip(le.classes_, le.transform(le.classes_)))


# %% Replace NaN with average value or 0
colReplaceAvg = [col for col in merge_cols if col not in colReplaceMost]
for col in colReplaceAvg[1:]:
    if col == 'Zip Code' or col == 'Lat Long':
        continue
    else:
        replaceNanWithAvg(col)


# %% Throw away the data columns I think is useless
merge_copy = merge_copy.drop('Count_x', axis=1)
merge_copy = merge_copy.drop('Count_y', axis=1)
merge_copy = merge_copy.drop('Country', axis=1)
merge_copy = merge_copy.drop('State', axis=1)
merge_copy = merge_copy.drop('City', axis=1)
merge_copy = merge_copy.drop('Zip Code', axis=1)
merge_copy = merge_copy.drop('Lat Long', axis=1)
merge_copy = merge_copy.drop('Latitude', axis=1)
merge_copy = merge_copy.drop('Longitude', axis=1)
merge_copy = merge_copy.drop('Count', axis=1)
merge_copy = merge_copy.drop('Quarter', axis=1)


# %% save after preprocessing data
print(merge_copy)
merge_copy.to_csv(os.path.join(dataSetPath, "merge_after_preprocessing.csv"))



