#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.linear_model import ElasticNet
import pandas as pd
import pytorch_tabnet
import torch
import numpy as np
import optuna
import multiprocessing as mp
#from pytorch_tabnet.tab_model import TabNetRegressor
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


# In[3]:


def cv_par(train_cv, test_cv, val_cv, lam, alpha, family):
    #print(set(train_cv[family]).intersection(set(test_cv[family])))
    #print(set(train_cv[family]).intersection(set(val_cv[family])))
    
    # create outcome
    y_train_cv = train_cv.outcome
    y_test_cv  = test_cv.outcome
    y_val_cv   = val_cv.outcome
    


    # Create an empty list to store the indices of non-binary columns
    scale_indices = []

    # Loop through the columns in the DataFrame
    for col in train_cv.columns:
        unique_values = train_cv[col].unique()  # Get unique values in the column
        if col!= 'outcome' and col!='Family_ID' and col !='smriPath' and col != 'Subject':
            scale_indices.append(train_cv.columns.get_loc(col))

    # scale train data
    train_dat_s = train_cv.copy()
    features = train_cv.iloc[:,scale_indices]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    train_cv.iloc[:,scale_indices] = features
    # scale test data
    test_dat_s = test_cv.copy()
    features = test_cv.iloc[:,scale_indices]
    features = scaler.transform(features.values)
    test_cv.iloc[:,scale_indices] = features
    # scale val data
    val_dat_s = val_cv.copy()
    features = val_cv.iloc[:,scale_indices]
    features = scaler.transform(features.values)
    val_cv.iloc[:,scale_indices] = features

    # create covariates
    x_train_cv = train_cv.drop([family, 'outcome', 'smriPath', 'Subject'], axis=1)
    x_test_cv  = test_cv.drop([family, 'outcome', 'smriPath', 'Subject'], axis=1)
    x_val_cv   = val_cv.drop([family, 'outcome', 'smriPath', 'Subject'], axis=1)
     
    
    # transforming to numpy arrays
    x_train_cv = x_train_cv.to_numpy()
    x_val_cv   = x_val_cv.to_numpy()
    x_test_cv  = x_test_cv.to_numpy()
    y_train_cv = y_train_cv.to_numpy().reshape(-1,1)
    y_val_cv   = y_val_cv.to_numpy().reshape(-1,1)
    y_test_cv  = y_test_cv.to_numpy().reshape(-1,1)
    
    # initiate the model
    model = ElasticNet(alpha=lam, l1_ratio=alpha, max_iter=10000, selection='random')
    
    # fit model
    model.fit(x_train_cv, y_train_cv)

    # determine accuracy for test set
    y_pred = model.predict(x_test_cv)
    
    mae = mean_absolute_error(y_test_cv, y_pred)
    ev = explained_variance_score(y_test_cv, y_pred)
    mse = mean_squared_error(y_test_cv, y_pred)
    r2 = r2_score(y_test_cv, y_pred)
    r, p = pearsonr(y_test_cv.squeeze(), y_pred.squeeze())
    
    return(mse)


# In[4]:


###################################
#''''''''''''''''''''''''''''''''''
# K-fold Hyperparameter Tuning
#''''''''''''''''''''''''''''''''''
###################################

### Function inputs: 
## folder (where the data is stored)
## rep (fold #)
## predictors (brain, cov, or full - which model to fit?)
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
    lst=["lamDict", "alphaDict"]
    if not any(item in lst for item in params):
        return f"Parameter dictionary must have at least one of the following elements: {*lst,}"
    
    def objective(trial):

        
        # set lambda
        if "lamDict" not in params:
            lam = 0.1
        else:
            lam = trial.suggest_float("lam", **params["lamDict"])
            
        # set alpha
        if "alphaDict" not in params:
            alpha = 0.5
        else:
            alpha  = trial.suggest_float("alpha", **params["alphaDict"])

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
        scores = [cv_par(train_cv, test_cv, val_cv, lam, alpha, family) for train, test, val in zip(train_folds, test_folds, val_folds)]
        
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
    output_path = os.path.join('CNN', f'tune_res{rep}.pkl')
    #joblib.dump(study, output_path)
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
    print(rep)
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
    # Create an empty list to store the indices of non-binary columns
    scale_indices = []

    # Loop through the columns in the DataFrame
    for col in train_cv.columns:
        unique_values = train_cv[col].unique()  # Get unique values in the column
        if col!= 'outcome' and col!='Family_ID' and col !='smriPath' and col != 'Subject':
            scale_indices.append(train_cv.columns.get_loc(col))


    # scale train data
    train_dat_s = train_cv.copy()
    features = train_cv.iloc[:,scale_indices]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    train_cv.iloc[:,scale_indices] = features
    # scale test data
    test_dat_s = test_cv.copy()
    features = test_cv.iloc[:,scale_indices]
    #scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    test_cv.iloc[:,scale_indices] = features
    # scale val data
    val_dat_s = val_cv.copy()
    features = val_cv.iloc[:,scale_indices]
    #scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    val_cv.iloc[:,scale_indices] = features

    # create covariates
    x_train_cv_o = train_cv.drop([family, 'outcome', 'smriPath', 'Subject'], axis=1)
    x_test_cv_o  = test_cv.drop([family, 'outcome', 'smriPath', 'Subject'], axis=1)
    x_val_cv_o   = val_cv.drop([family, 'outcome', 'smriPath', 'Subject'], axis=1)


    # hyper-parameter search for the fold's training set
    best_value, best_params = hypsearchCV(folder, rep, predictors=predictors, params=params, method=method, n_trials=n_trials, family=family, k = k_inner)
    print(best_params)

    # set hyper-parameters for model fitting
    # set lambda
    if "lamDict" not in params:
        lam = 0.1
    else:
        lam = best_params["lam"]

    # set alpha
    if "alphaDict" not in params:
        alpha = 0.5
    else:
        alpha  = best_params["alpha"]



    ## fit model with best parameters

    # transforming to numpy arrays
    x_train_cv_o = x_train_cv_o.to_numpy()
    x_val_cv_o   = x_val_cv_o.to_numpy()
    x_test_cv_o  = x_test_cv_o.to_numpy()
    y_train_cv_o = y_train_cv_o.to_numpy().reshape(-1,1)
    y_val_cv_o   = y_val_cv_o.to_numpy().reshape(-1,1)
    y_test_cv_o  = y_test_cv_o.to_numpy().reshape(-1,1)

    # initiate the model
    model = ElasticNet(alpha=lam, l1_ratio=alpha, max_iter=10000, selection='random')
    
    # fit model
    model.fit(x_train_cv_o, y_train_cv_o)

    # determine accuracy for test set
    y_pred = model.predict(x_test_cv_o)

    test_mae, test_ev, test_mse, test_r2, test_r, test_p = evaluate_reg(y_test_cv_o, y_pred)

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
    pickle.dump(model, open(filename, 'wb'))

    return(best_value, best_params, test_mae, test_ev, test_mse, test_r2, test_r, test_p)


# In[5]:


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
    #l = [fold_eval(folder, rep, predictors, params, method, model_name, n_trials, k_inner, family) for rep in range(k_outer*n_rep)]
        pool    = mp.Pool(processes=k_outer*n_rep)
        results = [pool.apply_async(fold_eval, args=(folder, rep, predictors, params, method, model_name, n_trials, k_inner, family)) for rep in range(k_outer*n_rep)]
        l       = [p.get() for p in results]
    
    # Unpack the results into separate variables
    best_value, best_params, test_mae, test_ev, test_mse, test_r2, test_r, test_p = zip(*l)
    
    
    #for diction in best_params:
    #    params_dict = mergeDictionary(params_dict, diction)
        
    params_best = pd.DataFrame.from_dict(best_params)
    params_best = pd.concat([params_best, pd.DataFrame({"mse":best_value})], axis=1)
        
    
    # evaluation information
    eval_data = pd.DataFrame({"R2":test_r2, "MSE":test_mse, "MAE": test_mae, "ev": test_ev, "r": test_r, "p": test_p})
        

    return(params_best, eval_data)


# In[6]:


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


# In[7]:


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




