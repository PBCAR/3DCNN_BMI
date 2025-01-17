#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import utils_tune as ut
import torch
import joblib
import numpy as np
import pandas as pd
import optuna
import math
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.model_selection import KFold




### CV (do we want repeated CV?)
train_dat = pd.read_csv("ssd/tr_886_rep_0.csv")
#train_dat = pd.read_csv("ssd/tr_48_rep_0.csv")
# idea: save each fold's train, test, and validation split to ssd/fold_#/datafilename

# prepare the cross-validation procedure
# prepare the cross-validation procedure
n_split = 5
n_rep = 3
cv = RepeatedKFold(n_splits=n_split, n_repeats=n_rep, random_state=1)
#train_dat = pd.read_csv("ssd/tr_48_rep_0.csv")
train=train_dat
family="Family_ID"
#target = train_dat.outcome
fold_Size = []

#for (split_ind, fold) in zip(cv.split(train[family].unique()), range(n_split*n_rep)):
#    train_ind = split_ind[0]
#    test_ind = split_ind[1]
#    fold_Size.append(len(train_ind))
for (split_ind, fold) in zip(cv.split(train[family].unique()), range(n_split*n_rep)):
    train_ind = split_ind[0]
    test_ind = split_ind[1]
    #fold_Size.append(len(train_ind))
    data_id = train[family].unique()

    # split into train/test set
    train_cv = train.loc[train[family].isin(data_id[train_ind]), :]
    test_cv  = train.loc[train[family].isin(data_id[test_ind]), :] 
    #fold_Size.append(train_cv.shape[0])
    # create validation set    

    val_id, test_id = train_test_split(test_cv[family].unique(), test_size=0.50, random_state=8)
    
    val_cv = test_cv.loc[test_cv[family].isin(val_id), :]
    #y_val_cv = y_test_cv.loc[x_test_cv[family].isin(val_id)]     
    test_cv = test_cv.loc[test_cv[family].isin(test_id)]  
    #x_test_cv = x_test_cv.loc[x_test_cv[family].isin(test_id), :]
    # save train, test, and validation set
    print(set(train_cv[family]).intersection(set(test_cv[family])))
    print(set(train_cv[family]).intersection(set(val_cv[family])))
    print(set(test_cv[family]).intersection(set(val_cv[family])))
    tr_size = train_cv.shape[0]
    print(tr_size)
    fold_Size.append(tr_size)
    #print(tr_size)
    file_name_tr = ("ssd/cv/tr_%d_rep_%d.csv" %(tr_size, fold))
    file_name_te = ("ssd/cv/te_%d_rep_%d.csv" %(tr_size, fold))
    file_name_va = ("ssd/cv/va_%d_rep_%d.csv" %(tr_size, fold))
    
    # save dataframe in ssd folder 
    train_cv.to_csv(file_name_tr, index=False)
    test_cv.to_csv(file_name_te, index=False)
    val_cv.to_csv(file_name_va, index=False)


print(fold_Size)
joblib.dump(fold_Size, "ssd/cv/fold_sizes.pkl")

