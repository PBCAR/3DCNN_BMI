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
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score, mean_absolute_error, explained_variance_score, mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pickle
import os
optuna.logging.set_verbosity(optuna.logging.WARNING)

predictors='cov'
family="Family_ID"
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
    
### covariate model

## Get best fold #

# Read in 'Covariate data CV evaluation with TPES.csv'
cv_evaluation_df = pd.read_csv('HandCraft/Covariate data CV evaluation with TPES.csv')

# Find row number with max of R2 column, name it best_fold
best_fold = cv_evaluation_df['MSE'].idxmin()
print(best_fold)

## Get model from best fold
filename = f"HandCraft/EN_cov_TPES{best_fold}.pkl"
with open(filename, 'rb') as file:
    model = pickle.load(file)


# Define the folder path where your data files are located
folder_path = '../../Tabnet/Tabnet_BMI/ssd_tabnet/'

for filename in os.listdir(folder_path):
    if 'tr' in filename and 'rep_0_' in filename and predictors in filename:
        print(folder_path+filename)
        train_cv=pd.read_csv(folder_path+filename)
    elif 'te' in filename and 'rep_0_' in filename and predictors in filename:
        print(folder_path+filename)
        test_cv=pd.read_csv(folder_path+filename)
    elif 'va' in filename and 'rep_0_' in filename and predictors in filename:
        print(folder_path+filename)
        val_cv=pd.read_csv(folder_path+filename)

test_cv = pd.concat([test_cv, val_cv], ignore_index=True)

# create outcome
y_test_cv_o  = test_cv.outcome


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
x_test_cv_o  = test_cv.drop([family, 'outcome'], axis=1)

## fit model with best parameters

# transforming to numpy arrays
x_test_cv_o  = x_test_cv_o.to_numpy()
y_test_cv_o  = y_test_cv_o.to_numpy()#.reshape(-1,1)

# determine accuracy for test set
y_pred = model.predict(x_test_cv_o)

test_mae, test_ev, test_mse, test_r2, test_r, test_p = evaluate_reg(y_test_cv_o, y_pred)

print(test_r2)
print(test_mse)
print(test_mae)
# Save scores
#r2score.append(test_r2)
#msescore.append(test_mse)
#maescore.append(test_mae)           
#mapescore.append(test_mape) 

# Create a dictionary with the evaluation metrics
metrics = {
    "test_mae": [test_mae],
    "test_ev": [test_ev],
    "test_mse": [test_mse],
    "test_r2": [test_r2],
    "test_r": [test_r],
    "test_p": [test_p]
}

# Convert the dictionary to a DataFrame
metrics_df = pd.DataFrame(metrics)

# Save the DataFrame to a CSV file
metrics_df.to_csv("HandCraft/evaluation_metrics_lockbox_cov_alt.csv", index=False)
