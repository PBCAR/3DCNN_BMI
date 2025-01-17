#!/usr/bin/env python
# coding: utf-8

# In[10]:


#exec(open('RF_Utils.py').read())

#import RF_Utils as ut
exec(open('RF_Utils.py').read())
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
                                        folder = "../../Tabnet/Tabnet_BMI/ssd_tabnet/",
                                        predictors = "full",
                                        params = dict_test,
                                        method = "TPES",
                                        model_name = "HandCraft/RF_full_TPES",
                                        n_trials = 150,
                                        family = "Family_ID",
                                        k_outer = 5,
                                        k_inner = 3)
eval_full_tpes.to_csv("HandCraft/Full data CV evaluation with TPES.csv")
param_full_tpes.to_csv("HandCraft/Full data hyper-parameter tuning with TPES.csv")

# covariates with Random Search
param_cov_tpes, eval_cov_tpes = cv_eval(tss = 880, 
                                        folder = "../../Tabnet/Tabnet_BMI/ssd_tabnet/", 
                                        predictors = "cov",
                                        params = dict_test, 
                                        method = "TPES", 
                                        model_name = "HandCraft/RF_cov_TPES",
                                        n_trials = 150, 
                                        family = "Family_ID", 
                                        k_outer = 5, 
                                        k_inner = 3)
eval_cov_tpes.to_csv("HandCraft/Covariate data CV evaluation with TPES.csv")
param_cov_tpes.to_csv("HandCraft/Covariate data hyper-parameter tuning with TPES.csv")


# brain with TPES
param_brain_tpes, eval_brain_tpes = cv_eval(tss = 880,
                                        folder = "../../Tabnet/Tabnet_BMI/ssd_tabnet/",
                                        predictors = "brain",
                                        params = dict_test,
                                        method = "TPES",
                                        model_name = "HandCraft/RF_brain_TPES",
                                        n_trials = 150,
                                        family = "Family_ID",
                                        k_outer = 5,
                                        k_inner = 3)
eval_brain_tpes.to_csv("HandCraft/Brain data CV evaluation with TPES.csv")
param_brain_tpes.to_csv("HandCraft/Brain data hyper-parameter tuning with TPES.csv")
#
