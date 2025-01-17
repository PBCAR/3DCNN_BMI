#!/usr/bin/env python
# coding: utf-8

# In[10]:


#exec(open('RF_Utils.py').read())

#import RF_Utils as ut
exec(open('RF_Utils_CNN.py').read())
# In[11]:


# Parmaters to tune for each fold
dict_test = {
        "ntreeDict": {
                "low": 10, 
                "high": 1000,
                "step": 10}, 
        "depthDict": { 
                "low": 2, 
                "high": 20, 
                 "step": 1},
        "nsamplesDict": {
                "low": 2,
                "high": 25,
                "step": 1}}


# In[12]:
# full with TPES
param_full_tpes, eval_full_tpes = cv_eval(tss = 880,
                                        folder = "../ssd_tabnet_cnn/",
                                        predictors = "full",
                                        params = dict_test,
                                        method = "TPES",
                                        model_name = "CNN/RF_full_TPES",
                                        n_trials = 150,
                                        family = "Family_ID",
                                        k_outer = 5,
                                        k_inner = 3)
eval_full_tpes.to_csv("CNN/Full data CV evaluation with TPES.csv")
param_full_tpes.to_csv("CNN/Full data hyper-parameter tuning with TPES.csv")


#covariate with TPES
param_cov_tpes, eval_cov_tpes = cv_eval(tss = 880,
                                        folder = "../ssd_tabnet_cnn/",
                                        predictors = "cov",
                                        params = dict_test,
                                        method = "TPES",
                                        model_name = "CNN/RF_cov_TPES",
                                        n_trials = 150,
                                        family = "Family_ID",
                                        k_outer = 5,
                                        k_inner = 3)
eval_cov_tpes.to_csv("CNN/Covariate data CV evaluation with TPES.csv")
param_cov_tpes.to_csv("CNN/Covariate data hyper-parameter tuning with TPES.csv")

#covariate with TPES
param_brain_tpes, eval_brain_tpes = cv_eval(tss = 880,
                                        folder = "../ssd_tabnet_cnn/",
                                        predictors = "brain",
                                        params = dict_test,
                                        method = "TPES",
                                        model_name = "CNN/RF_brain_TPES",
                                        n_trials = 150,
                                        family = "Family_ID",
                                        k_outer = 5,
                                        k_inner = 3)
eval_brain_tpes.to_csv("CNN/Brain data CV evaluation with TPES.csv")
param_brain_tpes.to_csv("CNN/Brain data hyper-parameter tuning with TPES.csv")

