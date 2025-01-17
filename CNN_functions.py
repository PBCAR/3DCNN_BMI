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


# In[28]:


###################################
#''''''''''''''''''''''''''''''''''
# K-fold Hyperparameter Tuning
#''''''''''''''''''''''''''''''''''
###################################

### Function inputs: 
## train (train set - features only)
## target (train set - outcome only)
## params (dictionary of parameters to tune with options)
## method (method for hyper-parameter tuning)
## n_trials (# of hyperparameter combinations to test, default=10)
## family (family ID column name from train set, default is None)
## k (# of CV folds, default=5)
### Function outputs:
## Best r2 value, best hyperparameter values

def hypsearchCV(train, params, method, cv_fold, n_trials=10, reload=False, family=None, k=5):
    
    ## check inputs
    
    # family
    if family is None:
        print("No family ID given, the observation ID will be used (i.e., assuming no family)")
        family = "id"
        train["id"] = range(0, train.shape[0])
    
    # method
    if method not in ["CmaEs", "Random", "TPES"]:
        return "Method needs to be one of the following: CmaEs, Random, TPES"
    
    # params
    lst=["lrDict", "bsDict"]
    if not any(item in lst for item in params):
        return f"Parameter dictionary must have at least one of the following elements: {*lst,}"
    
    def objective(trial):
        
        # set learning rate
        if "lrDict" not in params:
            lr = 2e-2
        else:
            lr = trial.suggest_float("lr", **params["lrDict"])

        # set batch size
        if "bsDict" not in params:
            bs = 32
        else:
            bs_pow = trial.suggest_int("bs", **params["bsDict"])
            bs = 2**bs_pow 

        print(bs)

        # set dropout
        if "pDict" not in params:
            p = 0.5
        else:
            p = trial.suggest_float("p", **params["pDict"])

        # set weight decay
        if "l2Dict" not in params:
            l2 = 0.01
        else:
            l2 = trial.suggest_float("l2", **params["l2Dict"])
        # conduct CV (Parallelized)
        
        cv = KFold(n_splits=k, shuffle=True, random_state=8)
        #if __name__ == '__main__':
        #    pool = mp.Pool(processes=k)
        #    results = [pool.apply_async(cv_par, args=(split_ind, train, target, lr, bs, family, fold)) for train_ind, test_ind in cv.split(train[family])]
        #    scores = [p.get() for p in results]
        scores = [cv_par(split_ind, train, lr, bs, p, l2, family, fold, cv_fold) for (split_ind, fold) in zip(cv.split(train[family].unique()), range(k))]
        # calculate/return average score across folds
        return np.mean(scores)
    
    # selecting the sampler based on requested method (consider pruning methods? can we do pruning for tabnet?)
    if method=="CmaEs":
        sampler = optuna.samplers.CmaEsSampler()

    elif method=="Random":
        sampler = optuna.samplers.RandomSampler()
        
    elif method=="TPES":
        sampler = optuna.samplers.TPESampler()
    
    path = ("results/cv/hyp_opt/fold_%d/" %(cv_fold))
    
    if not os.path.exists(path):
        os.makedirs(path)
    if (reload == False):
      study = optuna.create_study(direction="minimize", sampler = sampler)
    elif(reload == True):
      study = joblib.load(path+'tune_res.pkl')
    study.optimize(objective, n_trials=n_trials, timeout=432000, gc_after_trial=False)
    joblib.dump(study, path+'tune_res.pkl')

    # find best value
    value_b = study.best_value
    
    # find optimal hyper-parameters
    param_b = study.best_params
    
    #print updates
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    # return optimal hyper-parameters 
    return(value_b, param_b)


# In[29]:


def cv_par(split_ind, train, lr, bs, p, l2, family, fold, cv_fold):
    
    train_ind = split_ind[0]
    test_ind = split_ind[1]
    data_id = train[family].unique()

    # split into train/test set
    train_cv = train.loc[train[family].isin(data_id[train_ind]), :]    
    test_cv  = train.loc[train[family].isin(data_id[test_ind]), :] 

    # create validation set
    val_id, test_id = train_test_split(test_cv[family].unique(), test_size=0.50, random_state=8)
    
    val_cv = test_cv.loc[test_cv[family].isin(val_id), :]
       
    test_cv = test_cv.loc[test_cv[family].isin(test_id)]  
    
    path = ("ssd/cv/hyp_tune/fold_%d/" %(cv_fold))

    if not os.path.exists(path):
        os.makedirs(path)
    
    tr_size = train_cv.shape[0]
    file_name_tr = ("ssd/cv/hyp_tune/fold_%d/tr_%d_rep_%d.csv" %(cv_fold, tr_size, fold))
    file_name_te = ("ssd/cv/hyp_tune/fold_%d/te_%d_rep_%d.csv" %(cv_fold, tr_size, fold))
    file_name_va = ("ssd/cv/hyp_tune/fold_%d/va_%d_rep_%d.csv" %(cv_fold, tr_size, fold))
    
    # save dataframe in ssd folder 
    train_cv.to_csv(file_name_tr, index=False)
    test_cv.to_csv(file_name_te, index=False)
    val_cv.to_csv(file_name_va, index=False)
    
    tss = tr_size
    rep = fold
    nc = 1 
    bs = bs 
    lr = lr
    p = p
    l2 = l2
    no_cuda = False
    es = 200 
    es_va = 1 
    es_pat = 40
    path = ("results/cv/hyp_tune/fold_%d/" %(cv_fold))
    if not os.path.exists(path):
        os.makedirs(path) 
    ml = path 
    mt = 'AlexNet3D_Dropout_Regression' 
    ssd = ("ssd/cv/hyp_tune/fold_%d/" %(cv_fold))  
    predictor = 'smriPath'  
    scorename = 'outcome' 
    nw = 4 
    cr = 'reg'
    seed = 5
    
    # whether or not to use cuda
    cuda_avl = not no_cuda and torch.cuda.is_available()
    #cuda_avl = False
    # Set up configuration
    cfg = ut.Config(nc=nc, bs=bs, lr=lr, p=p, l2=l2, es=es,
                    es_va=es_va, es_pat=es_pat, ml=ml, mt=mt,
                    ssd=ssd, predictor=predictor, scorename=scorename, cuda_avl=cuda_avl,
                    nw=nw, cr=cr, tss=tss, rep=rep)

    if cuda_avl:
        torch.cuda.manual_seed(seed)

    # Update model location (ml) based on other config parameters
    cfg = ut.updateML(cfg)

    # train
    ut.generate_validation_model(cfg)

    # test
    mae, ev, mse, r2, r, pr = ut.evaluate_test_accuracy(cfg)
    # determine accuracy for test set
    return(mse)


# In[30]


# In[33]:



