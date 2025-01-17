#!/usr/bin/env python
# coding: utf-8

# In[ ]:

exec(open('Utils_lred.py').read())

# In[ ]:




##--- Nested CV Evaluation with Hyperparameter Tuning ---##

# Parmaters to tune for each fold
dict_test = {
        "lrDict": {
                "low": 0.0001, 
                "high": 0.05,
                "log": True}, 
        "wDict": { 
                "low": 0.0001, 
                "high": 0.01, 
                 "log": True},
        #"gDict": {
        #        "low": 1,
        #        "high": 2,
        #        "log": False},
        #"ssDict": {
        #        "low": 3,
        #        "high": 15,
        #        "q": 1},
        "NaDict": {
                "low": 8,
                "high": 64,
                "q": 4
        },
       "bsDict": {
                "low": 3,
                "high": 5,
                "step": 1
       }
}



# with Random Search
param_cov_tpes, eval_cov_tpes = cv_eval(tss = 880, 
                                        folder = "ssd_tabnet/", 
                                        predictors = "cov",
                                        params = dict_test, 
                                        method = "TPES", 
                                        model_name = "June24/Tabnet_cov_TPES",
                                        n_trials = 150, 
                                        family = "Family_ID", 
                                        k_outer = 5, 
                                        k_inner = 3)
eval_cov_tpes.to_csv("June24/cov data CV evaluation with TPES.csv")
param_cov_tpes.to_csv("June24/cov data hyper-parameter tuning with TPES.csv")

