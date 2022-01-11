# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 23:40:48 2022

@author: Eric
"""

import pandas as pd
from glob import glob
import os


# %%
# read all csv
dataSetPath = "dataset"
train_path = os.path.join(dataSetPath, "Train_IDs.csv")  # path to training data
test_path = os.path.join(dataSetPath, "Test_IDs.csv")    # path to testing data

files = glob(os.path.join(dataSetPath, "*.csv"))
data_csv = []
data_csv.append(train_path)
for csv in files:
    if ('IDs' not in csv) and ('sample' not in csv) and ('population' not in csv) and ('merge' not in csv):
        data_csv.append(csv)
  
print(data_csv)
df_list = [pd.read_csv(file) for file in data_csv]


# merge all csv
merge = df_list[0]
for df in df_list[1:]:
    merge = pd.merge(merge, df, how='right', on='Customer ID')

merge.to_csv(os.path.join(dataSetPath, "merge.csv")) # Save combined data to merge.csv

print(merge)