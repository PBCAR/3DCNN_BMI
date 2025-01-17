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
from CNN_functions import hypsearchCV, cv_par
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser(description='BMI prediction, fold splits')

parser.add_argument('--fold', default=0, type=int, help='')

args = parser.parse_args()


### CV (do we want repeated CV?)
cv_fold = args.fold

fold_sizes = joblib.load("ssd/cv/fold_sizes.pkl")
print(fold_sizes)
fold_size = fold_sizes[cv_fold]

train_cv = pd.read_csv("ssd/cv/tr_%d_rep_%d.csv" %(fold_size, cv_fold)) # read in fold data here after saving training set size and rep

# load parameters
path = ("results/cv/hyp_opt/fold_%d/" %(cv_fold))
study = joblib.load(path+'tune_res2.pkl')
print(len(study.trials))
# find best value
best_value = study.best_value

# find optimal hyper-parameters
best_params = study.best_params

print("best value on fold \# %d is %d" % (cv_fold, best_value))

tss = fold_size
rep = cv_fold
nc = 1 
bs = 2**best_params["bs"] 
lr = best_params["lr"] 
p  = best_params["p"]
l2 = best_params["l2"]
no_cuda = False
es = 200 
es_va = 1 
es_pat = 40 
ml = "results/cv/" 
mt = 'AlexNet3D_Dropout_Regression' 
ssd = "ssd/cv/"  
predictor = 'smriPath'  
scorename = 'outcome' 
nw = 4 
cr = 'reg'
    
# whether or not to use cuda
cuda_avl = not no_cuda and torch.cuda.is_available()
#cuda_avl = False
# Set up configuration
cfg = ut.Config(nc=nc, bs=bs, lr=lr, p=p, l2=l2, es=es,
                es_va=es_va, es_pat=es_pat, ml=ml, mt=mt,
                ssd=ssd, predictor=predictor, scorename=scorename, cuda_avl=cuda_avl,
                nw=nw, cr=cr, tss=tss, rep=rep)

if cuda_avl:
    torch.cuda.manual_seed(123)

# Update model location (ml) based on other config parameters
cfg = ut.updateML(cfg)

# train
ut.generate_validation_model(cfg)

# test (save dataframe with this info so we can summarize after)
mae, ev, mse, r2, r, pr = ut.evaluate_test_accuracy(cfg)
print("R2 on fold # %d is %d" % (fold, r2))


# In[36]:


dat_sum = pd.DataFrame({'mae': mae_vec, 'ev': ev_vec, 'mse': mse_vec, 'r2': r2_vec, 'r': r_vec, 'p': p_vec})
dat_sum.to_csv("results/results_summary_cv_%d.csv"%(cv_fold)) # add fold number to name
print(dat_sum)


# In[ ]:




