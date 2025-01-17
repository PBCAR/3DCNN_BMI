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
import argparse
#from CNN_functions import hypsearchCV, cv_par
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser(description='BMI prediction, fold splits')

parser.add_argument('--fold', default=0, type=int, help='')

parser.add_argument('--separable', action='store_true', help='')

parser.add_argument('--reload', action='store_true', help='')

args = parser.parse_args()

if args.separable:
    from CNN_functions_separable import hypsearchCV, cv_par
else:
    from CNN_functions import hypsearchCV, cv_par

### CV (do we want repeated CV?)
cv_fold = args.fold

fold_sizes = joblib.load("ssd/cv/fold_sizes.pkl")
print(fold_sizes)
fold_size = fold_sizes[cv_fold]


train_cv = pd.read_csv("ssd/cv/tr_%d_rep_%d.csv" %(fold_size, cv_fold)) # read in fold data here after saving training set size and rep

dict_test = {
        "lrDict": {
                "low": 0.0001,
                "high": 0.05,
                "log": True},
       "bsDict": {
                "low": 3,
                "high": 5,
                "step": 1},
       "pDict": {
                "low": 0.3,
                "high": 0.8,
                "log": False},
       "l2Dict": {
                "low": 0.0001,
                "high": 0.5,
                "log": True},

}

path = ("results/cv/hyp_opt/fold_%d/" %(cv_fold))
study = joblib.load(path+'tune_res2.pkl')
num_trials = len(study.trials)
print(num_trials)
print(args.reload)
n_trials = 75-num_trials
#n_trials = 2

# hyper-parameter search for the fold's training set
best_value, best_params = hypsearchCV(train_cv, params=dict_test, method="TPES", cv_fold=cv_fold, n_trials=n_trials, reload=args.reload, family="Family_ID", k = 3)
print("best value on fold \# %d is %d" % (cv_fold, best_value))



#path = ("results/cv/hyp_opt/fold_%d/" %(cv_fold))
#study = joblib.load(path+'tune_res.pkl')
#num_trials = len(study.trials)
#print(num_trials)
