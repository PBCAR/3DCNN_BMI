#!/usr/bin/env python
# coding: utf-8

# In[41]:


## Importing required libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import re


# In[42]:


# read in train, test, val set from CNN
train_dat = pd.read_csv("../../deep_learning/BMI/ssd_recovered/ssd/tr_886_rep_0.csv")
test_dat = pd.read_csv("../../deep_learning/BMI/ssd_recovered/ssd/te_886_rep_0.csv")
val_dat = pd.read_csv("../../deep_learning/BMI/ssd_recovered/ssd/va_886_rep_0.csv")

# read in tabular data
# Restricted HCP data
data1 = pd.read_csv("RESTRICTED_owensmax_2_8_2021_13_31_28.csv")
print(data1.shape) # 1206, 201

# Unrestricted HCP data
data2 = pd.read_csv("unrestricted_owensmax_2_2_2021_13_58_55.csv")
print(data2.shape) # 1206, 582


# In[43]:


# create outcome
data1 = data1.assign(outcome=data1.BMI)

## Creating final dataset

# List of wanted volumes
lst = ['FS_L_Cerebellum_Cort_Vol', 
       'FS_L_ThalamusProper_Vol', 
       'FS_L_Caudate_Vol', 
       'FS_L_Putamen_Vol', 
       'FS_L_Pallidum_Vol',
       'FS_L_Hippo_Vol',
       'FS_L_Amygdala_Vol',
       'FS_L_AccumbensArea_Vol',
       'FS_R_VentDC_Vol',
       'FS_R_Cerebellum_Cort_Vol',
       'FS_R_ThalamusProper_Vol',
       'FS_R_Caudate_Vol',
       'FS_R_Putamen_Vol',
       'FS_R_Pallidum_Vol',
       'FS_R_Hippo_Vol',
       'FS_R_Amygdala_Vol',
       'FS_R_AccumbensArea_Vol',
       'FS_R_VentDC_Vol'] # add total intercranial volume 



# Selecting columns from restricted and unrestricted
data_f = pd.concat([data1.Subject, data2.Gender, data1.Age_in_Yrs, data1.SSAGA_Educ, data1.SSAGA_Income, data2.FS_IntraCranial_Vol, data1.Family_ID, data2.filter(regex='Thck$|Area$',axis=1), data2[data2.columns.intersection(lst)], data1.outcome], axis=1)

# Creating Dummy for Gender
data_f = pd.get_dummies(data_f, columns=['Gender']).drop(["Gender_F"], axis=1)

# Drop those with missing values 
data_f_c = data_f.dropna()

print("Number of missing cases: %d" % (data_f.shape[0]-data_f_c.shape[0]))
print("Final data dimensions: %d, %d" % (data_f_c.shape[0], data_f_c.shape[1]))


# In[44]:


# combine data to get final train, test, validation sets

# train set
collect_dat = pd.DataFrame({"Subject":train_dat.Subject})
train_dat_c = collect_dat.merge(data_f_c, how='inner', on='Subject')
train_dat_c = train_dat_c.drop('Subject', axis=1)
print("final train data dimension with non-image removed: %d" % train_dat_c.shape[0])

# test set
collect_dat = pd.DataFrame({"Subject":test_dat.Subject})
test_dat_c = collect_dat.merge(data_f_c, how='inner', on='Subject')
test_dat_c = test_dat_c.drop('Subject', axis=1)
print("final test data dimension with non-image removed: %d" % test_dat_c.shape[0])

# validation set
collect_dat = pd.DataFrame({"Subject":val_dat.Subject})
val_dat_c = collect_dat.merge(data_f_c, how='inner', on='Subject')
val_dat_c = val_dat_c.drop('Subject', axis=1)
print("final val data dimension with non-image removed: %d" % val_dat_c.shape[0])


# In[45]:


## creating covariate and brain sets

# list of covariates
covs = ['Gender_M', 'Age_in_Yrs', 'SSAGA_Educ', 'SSAGA_Income', 'FS_IntraCranial_Vol']

# Split data into only brain variables
train_br = train_dat_c.drop(covs, axis=1)
test_br = test_dat_c.drop(covs, axis=1)
val_br = val_dat_c.drop(covs, axis=1)

# Split data into only covariates
train_cov = pd.concat([train_dat_c.Family_ID,train_dat_c[train_dat_c.columns.intersection(covs)], train_dat_c.outcome], axis=1)
test_cov = pd.concat([test_dat_c.Family_ID,test_dat_c[test_dat_c.columns.intersection(covs)], test_dat_c.outcome], axis=1)
val_cov = pd.concat([val_dat_c.Family_ID,val_dat_c[val_dat_c.columns.intersection(covs)], val_dat_c.outcome], axis=1)



# In[46]:


# Save datasets for future use
train_dat_c.to_csv("ssd_tabnet/tr_880_rep_0_full.csv", index=False)
test_dat_c.to_csv("ssd_tabnet/te_880_rep_0_full.csv", index=False)
val_dat_c.to_csv("ssd_tabnet/va_880_rep_0_full.csv", index=False)
train_br.to_csv("ssd_tabnet/tr_880_rep_0_brain.csv", index=False)
test_br.to_csv("ssd_tabnet/te_880_rep_0_brain.csv", index=False)
val_br.to_csv("ssd_tabnet/va_880_rep_0_brain.csv", index=False)
train_cov.to_csv("ssd_tabnet/tr_880_rep_0_cov.csv", index=False)
test_cov.to_csv("ssd_tabnet/te_880_rep_0_cov.csv", index=False)
val_cov.to_csv("ssd_tabnet/va_880_rep_0_cov.csv", index=False)


# In[47]:


import os
## now save the cv folds

# Define the folder path where data files are located
folder_path = '../../deep_learning/BMI/ssd_recovered/ssd/cv/' 

# Loop through the files in the folder
for f in range(15):
    for filename in os.listdir(folder_path):
        if 'tr' in filename and 'rep_%d.csv' %f in filename:
            train_cv=pd.read_csv(folder_path+filename)
            train_cv_br = train_br[train_br.Family_ID.isin(train_cv.Family_ID)]
            train_cv_cov = train_cov[train_cov.Family_ID.isin(train_cv.Family_ID)]
            train_cv_f = train_dat_c[train_dat_c.Family_ID.isin(train_cv.Family_ID)]
            
        elif 'te' in filename and 'rep_%d.csv' %f in filename:
            test_cv=pd.read_csv(folder_path+filename)
            print(test_cv.Family_ID)
            test_cv_br = train_br[train_br.Family_ID.isin(test_cv.Family_ID)]
            test_cv_cov = train_cov[train_cov.Family_ID.isin(test_cv.Family_ID)]
            test_cv_f = train_dat_c[train_dat_c.Family_ID.isin(test_cv.Family_ID)]
            
        elif 'va' in filename and 'rep_%d.csv' %f in filename:
            val_cv=pd.read_csv(folder_path+filename)
            val_cv_br = train_br[train_br.Family_ID.isin(val_cv.Family_ID)]
            val_cv_cov = train_cov[train_cov.Family_ID.isin(val_cv.Family_ID)]
            val_cv_f = train_dat_c[train_dat_c.Family_ID.isin(val_cv.Family_ID)]
    # Save datasets for future use
    fold_size = train_cv.shape[0]
    train_cv_f.to_csv("ssd_tabnet/cv/tr_%d_rep_%d_full.csv" %(fold_size, f), index=False)
    test_cv_f.to_csv("ssd_tabnet/cv/te_%d_rep_%d_full.csv" %(fold_size, f), index=False)
    val_cv_f.to_csv("ssd_tabnet/cv/va_%d_rep_%d_full.csv" %(fold_size, f), index=False)
    train_cv_br.to_csv("ssd_tabnet/cv/tr_%d_rep_%d_brain.csv" %(fold_size, f), index=False)
    test_cv_br.to_csv("ssd_tabnet/cv/te_%d_rep_%d_brain.csv" %(fold_size, f), index=False)
    val_cv_br.to_csv("ssd_tabnet/cv/va_%d_rep_%d_brain.csv" %(fold_size, f), index=False)
    train_cv_cov.to_csv("ssd_tabnet/cv/tr_%d_rep_%d_cov.csv" %(fold_size, f), index=False)
    test_cv_cov.to_csv("ssd_tabnet/cv/te_%d_rep_%d_cov.csv" %(fold_size, f), index=False)
    val_cv_cov.to_csv("ssd_tabnet/cv/va_%d_rep_%d_cov.csv" %(fold_size, f), index=False)


# In[49]:


## now save the nested cv folds
# Define the folder path where data files are located


# Loop through the files in the folder
for f in range(15):
    for k in range(3):
        folder_path = '../../deep_learning/BMI/ssd_recovered/ssd/cv/hyp_tune/fold_%d/' %f

        for filename in os.listdir(folder_path):
            if 'tr' in filename and 'rep_%d.csv' %k in filename:
                train_cv=pd.read_csv(folder_path+filename)
                train_cv_br = train_br[train_br.Family_ID.isin(train_cv.Family_ID)]
                train_cv_cov = train_cov[train_cov.Family_ID.isin(train_cv.Family_ID)]
                train_cv_f = train_dat_c[train_dat_c.Family_ID.isin(train_cv.Family_ID)]

            elif 'te' in filename and 'rep_%d.csv' %k in filename:
                test_cv=pd.read_csv(folder_path+filename)
                test_cv_br = train_br[train_br.Family_ID.isin(test_cv.Family_ID)]
                test_cv_cov = train_cov[train_cov.Family_ID.isin(test_cv.Family_ID)]
                test_cv_f = train_dat_c[train_dat_c.Family_ID.isin(test_cv.Family_ID)]

            elif 'va' in filename and 'rep_%d.csv' %k in filename:
                val_cv=pd.read_csv(folder_path+filename)
                val_cv_br = train_br[train_br.Family_ID.isin(val_cv.Family_ID)]
                val_cv_cov = train_cov[train_cov.Family_ID.isin(val_cv.Family_ID)]
                val_cv_f = train_dat_c[train_dat_c.Family_ID.isin(val_cv.Family_ID)]
        
        # Save datasets for future use
        fold_size = train_cv.shape[0]
        print(fold_size)
        fold_path_tab = "ssd_tabnet/cv/hyp_tune/fold_%d/" %f
        # Make Model Directory
        
        try:
            os.stat(fold_path_tab)
        except:
            os.mkdir(fold_path_tab)
        print(train_cv_cov.shape[0])
        train_cv_f.to_csv(fold_path_tab+"tr_%d_rep_%d_full.csv" %(fold_size, k), index=False)
        test_cv_f.to_csv(fold_path_tab+"te_%d_rep_%d_full.csv" %(fold_size, k), index=False)
        val_cv_f.to_csv(fold_path_tab+"va_%d_rep_%d_full.csv" %(fold_size, k), index=False)
        train_cv_br.to_csv(fold_path_tab+"tr_%d_rep_%d_brain.csv" %(fold_size, k), index=False)
        test_cv_br.to_csv(fold_path_tab+"te_%d_rep_%d_brain.csv" %(fold_size, k), index=False)
        val_cv_br.to_csv(fold_path_tab+"va_%d_rep_%d_brain.csv" %(fold_size, k), index=False)
        train_cv_cov.to_csv(fold_path_tab+"tr_%d_rep_%d_cov.csv" %(fold_size, k), index=False)
        test_cv_cov.to_csv(fold_path_tab+"te_%d_rep_%d_cov.csv" %(fold_size, k), index=False)
        val_cv_cov.to_csv(fold_path_tab+"va_%d_rep_%d_cov.csv" %(fold_size, k), index=False)


# In[ ]:




