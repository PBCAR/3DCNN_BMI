#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
cv_evaluation_df = pd.read_csv('CNN/Brain data CV evaluation with TPES.csv')

# Find row number with max of R2 column, name it best_fold
best_fold = cv_evaluation_df['MSE'].idxmin()
print(best_fold)

## Get model from best fold
filename = f"CNN/RF_brain_TPES5.pkl"
with open(filename, 'rb') as file:
    model = pickle.load(file)


# Define the folder path where your data files are located
folder_path = '../ssd_tabnet_cnn/'

for filename in os.listdir(folder_path):
    if 'te' in filename and 'brain_alt' in filename:
        test_cv=pd.read_csv(folder_path+filename)

# create outcome
y_test_cv_o  = test_cv.outcome

# create covariates
x_test_cv_o  = test_cv.drop([family, 'outcome', 'Subject', 'smriPath'], axis=1)

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
metrics_df.to_csv("CNN/evaluation_metrics_lockbox_brain_cnn_alt.csv", index=False)
