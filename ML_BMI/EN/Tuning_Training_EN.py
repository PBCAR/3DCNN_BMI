#!/usr/bin/env python
# coding: utf-8

# In[8]:


exec(open('EN_Utils.py').read())

# In[9]:


# Parmaters to tune for each fold
dict_test = {
        "lamDict": {
                "low": 0.001, 
                "high": 100,
                "log": True}, 
        "alphaDict": { 
                "low": 0.0001, 
                "high": 0.999, 
                 "log": True}}
        


# In[10]:

# cov with TPES
param_cov_tpes, eval_cov_tpes = cv_eval(tss = 880,
                                        folder = "../../Tabnet/Tabnet_BMI/ssd_tabnet/",
                                        predictors = "cov",
                                        params = dict_test,
                                        method = "TPES",
                                        model_name = "HandCraft/EN_cov_TPES",
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
                                        model_name = "HandCraft/EN_brain_TPES",
                                        n_trials = 150, 
                                        family = "Family_ID", 
                                        k_outer = 5, 
                                        k_inner = 3)
eval_brain_tpes.to_csv("HandCraft/Brain data CV evaluation with TPES.csv")
param_brain_tpes.to_csv("HandCraft/Brain data hyper-parameter tuning with TPES.csv")


# full with TPES
param_full_tpes, eval_full_tpes = cv_eval(tss = 880,
                                        folder = "../../Tabnet/Tabnet_BMI/ssd_tabnet/",
                                        predictors = "full",
                                        params = dict_test,
                                        method = "TPES",
                                        model_name = "HandCraft/EN_full_TPES",
                                        n_trials = 150,
                                        family = "Family_ID",
                                        k_outer = 5,
                                        k_inner = 3)
eval_full_tpes.to_csv("HandCraft/Full data CV evaluation with TPES.csv")
param_full_tpes.to_csv("HandCraft/Full data hyper-parameter tuning with TPES.csv")

# In[13]:


eval_brain_tpes.mean()


# In[14]:


eval_brain_tpes.std()




