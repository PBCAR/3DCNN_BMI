#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import KFold, RepeatedKFold
import pandas as pd
import pytorch_tabnet
import torch
import numpy as np
import optuna
import multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score, mean_absolute_error, explained_variance_score, mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pickle
import os
optuna.logging.set_verbosity(optuna.logging.WARNING)


# In[2]:


#####################################
#''''''''''''''''''''''''''''''''''''
# Evaluation of Regression-Type Model
#''''''''''''''''''''''''''''''''''''
#####################################

### Function inputs: 
## true (nx1 array of true y values)
## pred (nx1 array of predicted y values)
### Function outputs:
## r2 score, MSE, MAE, MAPE

def evaluate_reg(y_true, y_pred):
    
    mae = mean_absolute_error(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r, p = pearsonr(y_true.squeeze(), y_pred.squeeze())
    
    return(mae, ev, mse, r2, r, p)


# In[ ]:


def cv_par(train_cv, test_cv, val_cv, lr, w, na, bs, family):
    #print(set(train_cv[family]).intersection(set(test_cv[family])))
    #print(set(train_cv[family]).intersection(set(val_cv[family])))
    # create outcome
    y_train_cv = train_cv.outcome
    y_test_cv  = test_cv.outcome
    y_val_cv   = val_cv.outcome
    
    # create covariates
    x_train_cv = train_cv.drop([family, 'outcome'], axis=1)
    x_test_cv  = test_cv.drop([family, 'outcome'], axis=1)
    x_val_cv   = val_cv.drop([family, 'outcome'], axis=1)
     
     

        
    # transforming to numpy arrays
    x_train_cv = x_train_cv.to_numpy()
    x_val_cv   = x_val_cv.to_numpy()
    x_test_cv  = x_test_cv.to_numpy()
    y_train_cv = y_train_cv.to_numpy().reshape(-1,1)
    y_val_cv   = y_val_cv.to_numpy().reshape(-1,1)
    y_test_cv  = y_test_cv.to_numpy().reshape(-1,1)
    
    # initiate the model
    clf1_nopreproc = TabNetRegressor(optimizer_fn=torch.optim.Adam,
                        n_d = na,
                        n_a = na,
                        verbose=0,
                        optimizer_params=dict(lr=lr, weight_decay=w),
                        scheduler_params={"patience":7, # how to use learning rate scheduler
                                          "mode":'min',
                                          "factor":0.5},
                        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                        #scheduler_params={"step_size":ss, # how to use learning rate scheduler
                        #                  "gamma":g},
                        #scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        mask_type='entmax' # "sparsemax"
                        )
    
    # fit the model
    print(bs)
    print(bs/4)
    print(x_val_cv.shape) 
    clf1_nopreproc.fit(
        X_train=x_train_cv, y_train=y_train_cv,
        eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)],
        eval_name=['train', 'valid'],
        eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
        max_epochs=200,
        patience=40,
        batch_size=bs, 
        virtual_batch_size=bs/2,
        num_workers=0, # change to args.nw
        drop_last=True
        )

    # determine accuracy for test set
    y_pred = clf1_nopreproc.predict(x_test_cv)
    
    mae = mean_absolute_error(y_test_cv, y_pred)
    ev = explained_variance_score(y_test_cv, y_pred)
    mse = mean_squared_error(y_test_cv, y_pred)
    r2 = r2_score(y_test_cv, y_pred)
    r, p = pearsonr(y_test_cv.squeeze(), y_pred.squeeze())
    
    return(mse)


# In[ ]:


###################################
#''''''''''''''''''''''''''''''''''
# K-fold Hyperparameter Tuning
#''''''''''''''''''''''''''''''''''
###################################

### Function inputs: 
## folder (where train/test sets are stored)
## rep (fold #)
## predictors (brain, cov, full - which model to fit?)
## params (dictionary of parameters to tune with options)
## method (method for hyper-parameter tuning)
## n_trials (# of hyperparameter combinations to test, default=10)
## family (family ID column name from train set, default is None)
## k (# of CV folds, default=5)
### Function outputs:
## Best r2 value, best hyperparameter values

def hypsearchCV(folder, rep, predictors, params, method, n_trials=10, family=None, k=5):
    ## check inputs
    
    # method
    if method not in ["CmaEs", "Random", "TPES"]:
        return "Method needs to be one of the following: CmaEs, Random, TPES"
    
    
    # params
    lst=["lrDict", "wDict", "gDict", "ssDict", "bsDict"]
    if not any(item in lst for item in params):
        return f"Parameter dictionary must have at least one of the following elements: {*lst,}"
    
    def objective(trial):

        
        # set learning rate
        if "lrDict" not in params:
            lr = 2e-2
        else:
            lr = trial.suggest_float("lr", **params["lrDict"])
            
        # set weight decay
        if "wDict" not in params:
            w = 0.01
        else:
            w  = trial.suggest_float("w", **params["wDict"])
        
#        # set gamma
#        if "gDict" not in params:
#            g = 1.3
#        else:
#            g  = trial.suggest_float("g", **params["gDict"])
#            
#        # set step size
#        if "ssDict" not in params:
#            ss = 10
#        else:
#            ss = trial.suggest_discrete_uniform("ss", **params["ssDict"])
#            
        # set batch size
        if "bsDict" not in params:
            bs = 32
        else:
            bs_pow = trial.suggest_int("bs", **params["bsDict"])
            bs = 2**bs_pow 
            
        # set feature_dim = output_dim
        if "NaDict" not in params:
            na = int(8)
        else:
            na = int(trial.suggest_discrete_uniform("na", **params["NaDict"]))
    
        # conduct CV 
        
        # Define the folder path where data files are located
        folder_path = folder+'cv/hyp_tune/fold_%d/' %rep

        train_folds = []
        test_folds = []
        val_folds = []

        # Loop through the files in the folder
        for f in range(k):
            for filename in os.listdir(folder_path):
                if 'tr' in filename and 'rep_%d_' %f in filename and predictors in filename:
                    train_cv=pd.read_csv(folder_path+filename)
                    train_folds.append(train_cv)
                elif 'te' in filename and 'rep_%d_' %f in filename and predictors in filename:
                    test_cv=pd.read_csv(folder_path+filename)
                    test_folds.append(test_cv)
                elif 'va' in filename and 'rep_%d_' %f in filename and predictors in filename:
                    val_cv=pd.read_csv(folder_path+filename)
                    val_folds.append(val_cv)
        
        # read in data for each fold
        #if __name__ == '__main__':
        #    pool = mp.Pool(processes=k)
        #    results = [pool.apply_async(cv_par, args=(train_ind, test_ind, train, target, lr, ss, g, w, na, bs, family)) for train_ind, test_ind in cv.split(train[family])]
        #    scores = [p.get() for p in results]
        scores = [cv_par(train_cv, test_cv, val_cv, lr, w, na, bs, family) for train, test, val in zip(train_folds, test_folds, val_folds)]
        
        # calculate/return average score across folds
        return np.mean(scores)
    
    # selecting the sampler based on requested method (consider pruning methods? can we do pruning for tabnet?)
    if method=="CmaEs":
        sampler = optuna.samplers.CmaEsSampler(seed=123)

    elif method=="Random":
        sampler = optuna.samplers.RandomSampler(seed=123)
        
    elif method=="TPES":
        sampler = optuna.samplers.TPESampler(seed=123)
        
    study = optuna.create_study(direction="minimize", sampler = sampler)
    study.optimize(objective, n_trials=n_trials)
    output_path = os.path.join('June24', f'tune_res{rep}.pkl')
    # Save fitted models
    pickle.dump(study, open(output_path, 'wb'))
    
    # find best value
    value_b = study.best_value
    
    # find optimal hyper-parameters
    param_b = study.best_params
    print(param_b)
    # return optimal hyper-parameters 
    return(value_b, param_b)


# In[ ]:
###################################
#''''''''''''''''''''''''''''''''''
# Find optimal hyperparameters
# and evaluate model on a given 
# fold
#''''''''''''''''''''''''''''''''''
###################################

### Function inputs: 
## folder (where the train/test sets are stored)
## rep (fold #)
## predictors (brain, cov, or full for which model to fit)
## params (tuning parameter options)
## method (tuning search method)
## model_name (name that the model should be saved as)
## n_trials (how many hyperparameter combinations to test)
## k_inner (# of inner folds for nested CV hyperparameter tuning)
## family (Family_ID)
### Function outputs:
## The optimal hyperparameters
## The MSE achieved with optimal hyperparameters
## Evaluation metrics for the fold


def fold_eval(folder, rep, predictors, params, method, model_name, n_trials, k_inner, family):

    # Define the folder path where your data files are located
    folder_path = folder+'cv/' 

    for filename in os.listdir(folder_path):
        if 'tr' in filename and 'rep_%d_' %rep in filename and predictors in filename:
            print(folder_path+filename)
            train_cv=pd.read_csv(folder_path+filename)
        elif 'te' in filename and 'rep_%d_' %rep in filename and predictors in filename:
            print(folder_path+filename)
            test_cv=pd.read_csv(folder_path+filename)
        elif 'va' in filename and 'rep_%d_' %rep in filename and predictors in filename:
            print(folder_path+filename)
            val_cv=pd.read_csv(folder_path+filename)
    print(f"Intersection of train and test CV sets for rep '{rep}': {set(train_cv[family]).intersection(set(test_cv[family]))}")
    print(f"Intersection of train and val CV sets for rep '{rep}': {set(train_cv[family]).intersection(set(val_cv[family]))}")
    #print(set(train_cv[family]).intersection(set(test_cv[family])))
    #print(set(train_cv[family]).intersection(set(val_cv[family])))
    # create outcome
    y_train_cv_o = train_cv.outcome
    y_test_cv_o  = test_cv.outcome
    y_val_cv_o   = val_cv.outcome

    # create covariates
    x_train_cv_o = train_cv.drop([family, 'outcome'], axis=1)
    x_test_cv_o  = test_cv.drop([family, 'outcome'], axis=1)
    x_val_cv_o   = val_cv.drop([family, 'outcome'], axis=1)


    #df_merge = pd.merge(x_train_cv_o, x_test_cv_o, how='inner')
    #print(df_merge.shape)

    # hyper-parameter search for the fold's training set
    best_value, best_params = hypsearchCV(folder, rep, predictors=predictors, params=params, method=method, n_trials=n_trials, family=family, k = k_inner)
    print(best_params)
    # save information of hyper-parameter tuning
    #params_dict = mergeDictionary(params_dict, best_params)
    #tuning_value.append(best_value)

    # set hyper-parameters for model fitting
    # set learning rate
    if "lrDict" not in params:
        lr = 2e-2
    else:
        lr = best_params["lr"]

    # set weight decay
    if "wDict" not in params:
        w = 0.01
    else:
        w  = best_params["w"]

#    # set gamma
#    if "gDict" not in params:
#        g = 1.3
#    else:
#        g  = best_params["g"]
#
##    # set step size
#    if "ssDict" not in params:
#        ss = 3
#    else:
#        ss = best_params["ss"]

    # set batch size
    if "bsDict" not in params:
        bs = 32
    else:
        bs_pow = best_params["bs"]
        bs = 2**bs_pow

    # set feature_dim = output_dim
    if "NaDict" not in params:
        na = int(8)
    else:
        na = int(best_params["na"])

    ## fit model with best parameters

    # transforming to numpy arrays
    x_train_cv_o = x_train_cv_o.to_numpy()
    x_val_cv_o   = x_val_cv_o.to_numpy()
    x_test_cv_o  = x_test_cv_o.to_numpy()
    y_train_cv_o = y_train_cv_o.to_numpy().reshape(-1,1)
    y_val_cv_o   = y_val_cv_o.to_numpy().reshape(-1,1)
    y_test_cv_o  = y_test_cv_o.to_numpy().reshape(-1,1)

    # initiate the model
    clf1_nopreproc = TabNetRegressor(optimizer_fn=torch.optim.Adam,
                       n_d = na,
                       n_a = na,
                       verbose=0,
                       scheduler_params={"patience":7, # how to use learning rate scheduler
                                         "mode":'min',
                                         "factor":0.5},
                       scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                       optimizer_params=dict(lr=lr, weight_decay=w),
                       #scheduler_params={"step_size":ss, # how to use learning rate scheduler
                       #                  "gamma":g},
                       #scheduler_fn=torch.optim.lr_scheduler.StepLR,
                       mask_type='entmax' # "sparsemax"
                      )


    # fit model to training fold 
    clf1_nopreproc.fit(
            X_train=x_train_cv_o, 
            y_train=y_train_cv_o,
            eval_set=[(x_train_cv_o, y_train_cv_o), (x_val_cv_o, y_val_cv_o)],
            eval_name=['train', 'valid'],
            eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
            max_epochs=200,
            patience=40,
            batch_size=bs, 
            virtual_batch_size=bs/2,
            num_workers=0,
            drop_last=True
            )

    ## evaluate predictions for test set
    preds = clf1_nopreproc.predict(x_test_cv_o)

    test_mae, test_ev, test_mse, test_r2, test_r, test_p = evaluate_reg(y_test_cv_o, preds)

    print(test_r2)
    print(test_mse)
    print(test_mae)
    print(best_params)
    print(best_value)
    # Save scores
    #r2score.append(test_r2)
    #msescore.append(test_mse)
    #maescore.append(test_mae)           
    #mapescore.append(test_mape) 

    # Save fitted models
    filename = '%s%d.pkl' % (model_name, rep)
    pickle.dump(clf1_nopreproc, open(filename, 'wb'))

    return(best_value, best_params, test_mae, test_ev, test_mse, test_r2, test_r, test_p)


# In[ ]:


###################################
#''''''''''''''''''''''''''''''''''
# K-fold Evaluation with Nested CV
# for Hyperparameter Tuning
#''''''''''''''''''''''''''''''''''
###################################

### Function inputs: 
## tss (training set size of full training set)
## folder (where training/test sets are stored)
## predictors (brain, cov, full - which model to fit?)
## params (dictionary of parameters to tune with options)
## method (method for hyper-parameter tuning)
## model_name (what to name saved models (will end in fold #))
## n_trials (# of hyperparameter combinations to test, default=10)
## family (family ID column name from train set, default is None)
## k_outer (# of CV folds for evaluation in outer loop, default=5)
## k_inner (# of CV folds for parameter tuning in inner loop, default=5)
### Function outputs:
## Two datasets are returned - 
## 1. The optimal parameters and best value from tuning each fold
## 2. The evaluation of each fold (r2, mse, mae, mape)

def cv_eval(tss, folder, predictors, params, method, model_name, n_trials=10, family=None, k_outer=5, k_inner=5, n_rep=3):
    
    # check family input
    if family is None:
        print("No family ID given, the observation ID will be used (i.e., assuming no family)")
        family = "id"
        train_f["id"] = range(tss)
    

    
    #l = [fold_eval(split_ind, train_f, target_f, params, method, model_name, n_trials, k_inner, fold, family) for (split_ind, fold) in zip(cv.split(train_f[family].unique()), range(k_outer*n_rep))]
    
    
    if __name__ == '__main__':
        pool    = mp.Pool(processes=k_outer*n_rep)
        results = [pool.apply_async(fold_eval, args=(folder, rep, predictors, params, method, model_name, n_trials, k_inner, family)) for rep in range(k_outer*n_rep)]
        l       = [p.get() for p in results]
    
    # Unpack the results into separate variables
    best_value, best_params, test_mae, test_ev, test_mse, test_r2, test_r, test_p = zip(*l)
    
    
    #for diction in best_params:
    #    params_dict = mergeDictionary(params_dict, diction)
        
    params_best = pd.DataFrame.from_dict(best_params)
    params_best = pd.concat([params_best, pd.DataFrame({"mse":best_value})], axis=1)
        
    
    #for train_ind, test_ind in cv.split(train_f[family].unique()):
    # hyper-parameter information
    #params_data          = pd.DataFrame.from_dict(params_dict)
    #params_data["Value"] = tuning_value
    
    # evaluation information
    eval_data = pd.DataFrame({"R2":test_r2, "MSE":test_mse, "MAE": test_mae, "ev": test_ev, "r": test_r, "p": test_p})
        

    return(params_best, eval_data)


# In[ ]:


#######################################
#''''''''''''''''''''''''''''''''''''''
# Feature Importance Plots for K Folds
#''''''''''''''''''''''''''''''''''''''
#######################################

### Function inputs: 
## model_file (file name for each model (must end in fold #))
## feat_list (list of features used to train the models)
## image_file (desired name for images when saving)
## k (number of folds/number of models fit)
## n_trials (# of hyperparameter combinations to test, default=10)
## n_feat (include the top n_feat features in the plot)
### Function outputs:
## A feature importance plot will be saved for each fold 


def feat_imp(model_file, feat_list, image_file, k, n_feat):
    for i in range(k):
        
        # load model
        with open("%s%d.pkl" % (model_file, i+1), 'rb') as file_name:
            new_clf = pickle.load(file_name)
            
        # finding and sorting feature importance
        feat_importances = new_clf.feature_importances_
        indices = np.argsort(feat_importances)
        
        # check that length of feature list = length of features in model
        print(len(feat_importances)==len(feat_list))
        
        # save plot of top n_feat

        plt.figure(figsize=(10, 8))
        plt.title("Feature importances")
        plt.barh(range(len(feat_importances[0:n_feat])), feat_importances[indices[0:n_feat]],
                   color="blue", align="center")

        # If you want to define your own labels,
        # change indices to a list of labels on the following line.
        plt.yticks(range(len(feat_importances[0:n_feat])), [feat_list[idx] for idx in indices[0:n_feat]], rotation=45)
        plt.ylim([-1, len(feat_importances[0:n_feat])])
        plt.savefig("%s%d.png" % (image_file, i+1), dpi=300, bbox_inches = "tight")
        plt.close()


# In[ ]:


#######################################
#''''''''''''''''''''''''''''''''''''''
# Lockbox Predictions for K Folds
#''''''''''''''''''''''''''''''''''''''
#######################################

### Function inputs: 
## model_file (file name for each model (must end in fold #))
## test_dat (the observations used for prediction - x only)
## pred_file (desired name for prediction data when saving)
## k (number of folds/number of models fit)
### Function outputs:
## A nxk dataset will be saved for with predictions 
## on lockbox from each fold 

def lockbox_pred(model_file, test_dat, pred_file, k):
     
    preds_all=pd.DataFrame()
    for i in range(k):
        
        # load model
        with open("%s%d.pkl" % (model_file, i+1), 'rb') as file_name:
                new_clf = pickle.load(file_name)
                
        # make predictions on lockbox
        preds = new_clf.predict(test_dat.to_numpy())
        
        preds_all = pd.concat([preds_all, pd.DataFrame(preds)], axis=1)
    
    preds_all.columns = np.arange(1, k+1, 1).tolist()
    preds_all.to_csv(pred_file)
    return(preds_all)


# In[ ]:




