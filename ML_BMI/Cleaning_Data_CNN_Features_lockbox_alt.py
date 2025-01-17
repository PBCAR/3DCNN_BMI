#!/usr/bin/env python
# coding: utf-8

# In[48]:


## Importing required libraries
import utils_tune as ut
import torch
import torch.nn as nn
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import re
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import nibabel as nib
from  models_tune import AlexNet3D_Dropout, AlexNet3D_Dropout_Regression #### note to self - added Regression model
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, explained_variance_score, mean_squared_error, r2_score
from dataclasses import dataclass
from scipy.stats import pearsonr

# In[49]:


# read in train, test, val set from CNN
train_dat = pd.read_csv("../deep_learning/BMI/ssd_recovered/ssd/tr_886_rep_0.csv")
test_dat = pd.read_csv("../deep_learning/BMI/ssd_recovered/ssd/te_886_rep_0.csv")
val_dat = pd.read_csv("../deep_learning/BMI/ssd_recovered/ssd/va_886_rep_0.csv")
#val_dat = pd.read_csv("../deep_learning/BMI/ssd/va_886_rep_0.csv")

# read in tabular data
# Restricted HCP data
data1 = pd.read_csv("../Tabnet/RESTRICTED_owensmax_2_8_2021_13_31_28.csv")
print(data1.shape) # 1206, 201

# Unrestricted HCP data
data2 = pd.read_csv("../Tabnet/unrestricted_owensmax_2_2_2021_13_58_55.csv")
print(data2.shape) # 1206, 582


# In[50]:


# create outcome
data1 = data1.assign(outcome=data1.BMI)

## Creating final dataset

# Selecting columns from restricted and unrestricted
data_f = pd.concat([data1.Subject, data2.Gender, data1.Age_in_Yrs, data1.SSAGA_Educ, data1.SSAGA_Income, data2.FS_IntraCranial_Vol, data1.Family_ID, data1.outcome], axis=1)

# Creating Dummy for Gender
data_f = pd.get_dummies(data_f, columns=['Gender']).drop(["Gender_F"], axis=1)

# Drop those with missing values 
data_f_c = data_f.dropna()

print("Number of missing cases: %d" % (data_f.shape[0]-data_f_c.shape[0]))
print("Final data dimensions: %d, %d" % (data_f_c.shape[0], data_f_c.shape[1]))


# In[52]:


# train set
collect_dat = pd.DataFrame({"Subject":train_dat.Subject})
train_dat_c = collect_dat.merge(data_f_c, how='inner', on='Subject')
train_dat_c = train_dat_c.merge(train_dat.loc[:,['smriPath', 'Subject']], how='inner', on='Subject')
#train_dat_c = train_dat_c.drop('Subject', axis=1)
print(train_dat_c['Subject'].dtypes)
print("final train data dimension with non-image removed: %d" % train_dat_c.shape[0])

# test set
collect_dat = pd.DataFrame({"Subject":test_dat.Subject})
test_dat_c = collect_dat.merge(data_f_c, how='inner', on='Subject')
test_dat_c = test_dat_c.merge(test_dat.loc[:,['smriPath', 'Subject']], how='inner', on='Subject')
#test_dat_c = test_dat_c.drop('Subject', axis=1)
print("final test data dimension with non-image removed: %d" % test_dat_c.shape[0])

# validation set
collect_dat = pd.DataFrame({"Subject":val_dat.Subject})
val_dat_c = collect_dat.merge(data_f_c, how='inner', on='Subject')
val_dat_c = val_dat_c.merge(val_dat.loc[:,['smriPath', 'Subject']], how='inner', on='Subject')
#val_dat_c = val_dat_c.drop('Subject', axis=1)
print("final val data dimension with non-image removed: %d" % val_dat_c.shape[0])

# add validation and test set
test_dat_c = pd.concat([test_dat_c, val_dat_c], ignore_index=True)
print(test_dat_c.shape)

# In[53]:


## creating covariate and brain sets

# list of covariates
covs = ['Gender_M', 'Age_in_Yrs', 'SSAGA_Educ', 'SSAGA_Income', 'FS_IntraCranial_Vol']

# Split data into only brain variables
#train_br = train_dat_c.drop(covs, axis=1)
#test_br = test_dat_c.drop(covs, axis=1)
#val_br = val_dat_c.drop(covs, axis=1)

# Split data into only covariates
#train_cov = pd.concat([train_dat_c.Family_ID,train_dat_c[train_dat_c.columns.intersection(covs)], train_dat_c.outcome], axis=1)
#test_cov = pd.concat([test_dat_c.Family_ID,test_dat_c[test_dat_c.columns.intersection(covs)], test_dat_c.outcome], axis=1)
#val_cov = pd.concat([val_dat_c.Family_ID,val_dat_c[val_dat_c.columns.intersection(covs)], val_dat_c.outcome], axis=1)


# In[54]:


# Save datasets for future use
#train_dat_c.to_csv("ssd_tabnet_cnn/tr_880_rep_0_full.csv", index=False)
#test_dat_c.to_csv("ssd_tabnet_cnn/te_880_rep_0_full.csv", index=False)
#val_dat_c.to_csv("ssd_tabnet_cnn/va_880_rep_0_full.csv", index=False)
#train_br.to_csv("ssd_tabnet_cnn/tr_880_rep_0_brain.csv", index=False)
#test_br.to_csv("ssd_tabnet_cnn/te_880_rep_0_brain.csv", index=False)
#val_br.to_csv("ssd_tabnet_cnn/va_880_rep_0_brain.csv", index=False)
#train_cov.to_csv("ssd_tabnet_cnn/tr_880_rep_0_cov.csv", index=False)
#test_cov.to_csv("ssd_tabnet_cnn/te_880_rep_0_cov.csv", index=False)
#val_cov.to_csv("ssd_tabnet_cnn/va_880_rep_0_cov.csv", index=False)


# In[58]:


def extract_features_from_cnn(cfg, mode, rep, folder):
    def conv3d_output_hook(module, input, output):
        conv3d_output_hook.conv3d_output = output

    # Read in image data
    dset = ut.MRIDataset(cfg, mode)

    # Load model
    # find fold with lowest MSE
    #path = ("folder")
    cv_summary = pd.read_csv(folder+"cv_summary.csv")
    best_fold = cv_summary['mse_te'].idxmin()

    # Retrieve the directory from the best fold row
    best_directory = cv_summary.loc[best_fold, 'Directory']
    print(best_directory)
    model_state_path = f"{folder}{best_directory}/model_state_dict.pt"
    print(model_state_path)
    #folder_path = folder
            
    net = ut.AlexNet3D_Dropout_Regression(num_classes=1)
    model = torch.nn.DataParallel(net)
    net = ut.load_net_weights2(model, model_state_path)
    net.eval()
    print(net)
    handle = net.module.features[17].register_forward_hook(conv3d_output_hook)

    # Iterate over dataloader batches
    features = []
    subjects = []
    for _, data in enumerate(dset):
        inputs, labels, sub_id = data
        #print(sub_id)
        if labels:
            inputs = torch.unsqueeze(torch.tensor(inputs), 1)
            #print(X.shape)
            inputs = Variable(inputs.cuda())
            #inputs, labels = Variable(torch.tensor(inputs)), Variable(torch.tensor(labels))
            #print(inputs)
            #print(labels)
            #print(sub_id)
            # Forward pass
            outputs = net(inputs)
            print(sub_id)
            print(outputs)
            xp = conv3d_output_hook.conv3d_output
            print(xp)
            x = xp.view(xp.size(0), -1)
            features.append(x.cpu().detach().numpy().squeeze())
            subjects.append(sub_id)
        else:
            print(f"Empty result for index {sub_id}. Skipping this observation.")
    handle.remove()
    features_df = pd.DataFrame(features)
    # Add subjects to the DataFrame
    features_df['Subject'] = subjects
    features_df['Subject'] = features_df['Subject'].astype(np.int64)
    return features_df


# In[ ]:





# In[55]:


import os


# save the full train set/test set/val set with cnn features
## Define the folder path where data files are located
f=0 # change back to zero
folder_path = '../deep_learning/BMI/ssd_recovered/ssd/'  # change back to not cv/
## Loop through the files in the folder
fold_size = 886 # change back to 886
#fold_sizes = joblib.load(folder_path+"/fold_sizes.pkl")
#fold_size = fold_sizes[5]
#if 'tr' in filename and 'rep_%d' %f in filename:
cfg = ut.Config(ssd='../deep_learning/BMI/ssd_recovered/ssd/', predictor='smriPath', scorename='outcome', cuda_avl=True,
                cr='reg', tss=fold_size, rep=f)
filename = 'tr_%d_rep_%d.csv'%(fold_size,f)
train_cv=pd.read_csv(folder_path+filename)
# get the image data
train_cv_im = extract_features_from_cnn(cfg, 'tr', f, '../deep_learning/BMI/results/cv/')
#train_cv_im['Subject'] = train_cv['Subject']
# Merge the DataFrames based on the common column
train_cv_im = pd.merge(train_cv_im, train_dat_c.Subject, how='inner', on='Subject')
#train_cv_im = train_cv_im[train_cv.Family_ID.isin(train_dat_c.Family_ID)]
# Sort train_cv_cov based on the order of 'Family_ID' in train_cv
train_cv_cov = train_dat_c[train_dat_c.Subject.isin(train_cv.Subject)]
train_cv_cov = pd.merge(train_cv_cov, train_cv.Subject, how='inner', on='Subject')
#train_cv_cov = train_dat_c[train_dat_c.Family_ID.isin(train_cv.Family_ID)]
train_cv_im = train_cv_im.reset_index(drop=True)
train_cv_cov = train_cv_cov.reset_index(drop=True)
train_cv_f = pd.concat([train_cv_cov, train_cv_im], axis=1)
print(train_cv_f.shape)
train_cv_br = train_cv_f.drop(covs, axis=1)
print(train_cv_br.shape)
print(train_cv_cov.shape)
filename = 'combined_%d_rep_%d.csv'%(fold_size,f)
test_cv=pd.read_csv(folder_path+filename)
# get the image data
test_cv_im = extract_features_from_cnn(cfg, 'combined', f, '../deep_learning/BMI/results/cv/')
print(test_cv_im.shape)
#test_cv_im['Subject'] = test_cv['Subject']
#test_cv_im = test_cv_im[test_cv.Family_ID.isin(train_dat_c.Family_ID)]
#test_cv_cov = train_dat_c[train_dat_c.Family_ID.isin(test_cv.Family_ID)]
# Merge the DataFrames based on the common column
test_cv_im = pd.merge(test_cv_im, test_dat_c.Subject, how='inner', on='Subject')
print(test_cv_im.shape)
#train_cv_im = train_cv_im[train_cv.Family_ID.isin(train_dat_c.Family_ID)]
# Sort train_cv_cov based on the order of 'Family_ID' in train_cv
#test_cv_cov = train_dat_c[train_dat_c.Subject.isin(test_cv.Subject)]
#test_cv_cov = test_cv_c
test_cv_cov = pd.merge(test_dat_c, test_cv.Subject, how='inner', on='Subject')
test_cv_im = test_cv_im.reset_index(drop=True)
test_cv_cov = test_cv_cov.reset_index(drop=True)
test_cv_f = pd.concat([test_cv_cov, test_cv_im], axis=1)
test_cv_br = test_cv_f.drop(covs, axis=1)
print(test_cv_br.columns)
#
filename = 'va_%d_rep_%d.csv'%(fold_size,f)
va_cv=pd.read_csv(folder_path+filename)
# get the image data
va_cv_im = extract_features_from_cnn(cfg, 'va', f, '../deep_learning/BMI/results/cv/')
#va_cv_im['Subject'] = va_cv['Subject']    
#va_cv_im = va_cv_im[va_cv.Family_ID.isin(train_dat_c.Family_ID)]
#va_cv_cov = train_dat_c[train_dat_c.Family_ID.isin(va_cv.Family_ID)]
# Merge the DataFrames based on the common column
va_cv_im = pd.merge(va_cv_im, val_dat_c.Subject, how='inner', on='Subject')
#train_cv_im = train_cv_im[train_cv.Family_ID.isin(train_dat_c.Family_ID)]
# Sort train_cv_cov based on the order of 'Family_ID' in train_cv
#va_cv_cov = train_dat_c[train_dat_c.Subject.isin(va_cv.Subject)]
va_cv_cov = pd.merge(val_dat_c, va_cv.Subject, how='inner', on='Subject') 
va_cv_im = va_cv_im.reset_index(drop=True)
va_cv_cov = va_cv_cov.reset_index(drop=True)
va_cv_f = pd.concat([va_cv_cov, va_cv_im], axis=1)
va_cv_br = va_cv_f.drop(covs, axis=1)
# Save datasets for future use
fold_size = train_cv.shape[0]
train_cv_f.to_csv("ssd_tabnet_cnn/tr_%d_rep_%d_full_alt.csv" %(fold_size, 0), index=False)
test_cv_f.to_csv("ssd_tabnet_cnn/te_%d_rep_%d_full_alt.csv" %(fold_size, 0), index=False)
va_cv_f.to_csv("ssd_tabnet_cnn/va_%d_rep_%d_full_alt.csv" %(fold_size, 0), index=False)
train_cv_br.to_csv("ssd_tabnet_cnn/tr_%d_rep_%d_brain_alt.csv" %(fold_size, 0), index=False)
test_cv_br.to_csv("ssd_tabnet_cnn/te_%d_rep_%d_brain_alt.csv" %(fold_size, 0), index=False)
va_cv_br.to_csv("ssd_tabnet_cnn/va_%d_rep_%d_brain_alt.csv" %(fold_size, 0), index=False)
train_cv_cov.to_csv("ssd_tabnet_cnn/tr_%d_rep_%d_cov_alt.csv" %(fold_size, 0), index=False)
test_cv_cov.to_csv("ssd_tabnet_cnn/te_%d_rep_%d_cov_alt.csv" %(fold_size, 0), index=False)
va_cv_cov.to_csv("ssd_tabnet_cnn/va_%d_rep_%d_cov_alt.csv" %(fold_size, 0), index=False)
del train_cv_cov
del train_cv_br
del train_cv_f
del test_cv_cov
del test_cv_br
del test_cv_f
del va_cv_cov
del va_cv_br
del va_cv_f














## now save the cv folds

## Define the folder path where data files are located
##folder_path = '../deep_learning/BMI/ssd/cv/' 
#### Loop through the files in the folder
##fold_sizes = joblib.load(folder_path+"/fold_sizes.pkl")
##for f in range(15):
#    fold_size = fold_sizes[f]
#    #if 'tr' in filename and 'rep_%d' %f in filename:
#    cfg = ut.Config(ssd='../deep_learning/BMI/ssd/cv/', predictor='smriPath', scorename='outcome', cuda_avl=True,
#                    cr='reg', tss=fold_size, rep=f)
#    filename = 'tr_%d_rep_%d.csv'%(fold_size,f)
#    train_cv=pd.read_csv(folder_path+filename)
#    # get the image data
#    train_cv_im = extract_features_from_cnn(cfg, 'tr', f, '../deep_learning/BMI/results/cv/')
#    train_cv_im['Subject'] = train_cv['Subject']
#    # Merge the DataFrames based on the common column
#    train_cv_im = pd.merge(train_cv_im, train_dat_c.Subject, how='inner', on='Subject')
#    #train_cv_im = train_cv_im[train_cv.Family_ID.isin(train_dat_c.Family_ID)]
#    # Sort train_cv_cov based on the order of 'Family_ID' in train_cv
#    train_cv_cov = train_dat_c[train_dat_c.Subject.isin(train_cv.Subject)]
#    train_cv_cov = pd.merge(train_cv_cov, train_cv.Subject, how='inner', on='Subject')
#    #train_cv_cov = train_dat_c[train_dat_c.Family_ID.isin(train_cv.Family_ID)]
#    train_cv_im = train_cv_im.reset_index(drop=True)
#    train_cv_cov = train_cv_cov.reset_index(drop=True)
#    train_cv_f = pd.concat([train_cv_cov, train_cv_im], axis=1)
#    print(train_cv_f.shape)
#    train_cv_br = train_cv_f.drop(covs, axis=1)
#    print(train_cv_br.shape)
#    print(train_cv_cov.shape)
#    filename = 'te_%d_rep_%d.csv'%(fold_size,f)
#
#
#    test_cv=pd.read_csv(folder_path+filename)
#    # get the image data
#    test_cv_im = extract_features_from_cnn(cfg, 'te', f, '../deep_learning/BMI/results/cv/')
#    test_cv_im['Subject'] = test_cv['Subject']
#    #test_cv_im = test_cv_im[test_cv.Family_ID.isin(train_dat_c.Family_ID)]
#    #test_cv_cov = train_dat_c[train_dat_c.Family_ID.isin(test_cv.Family_ID)]
#    # Merge the DataFrames based on the common column
#    test_cv_im = pd.merge(test_cv_im, train_dat_c.Subject, how='inner', on='Subject')
#    #train_cv_im = train_cv_im[train_cv.Family_ID.isin(train_dat_c.Family_ID)]
#    # Sort train_cv_cov based on the order of 'Family_ID' in train_cv
#    test_cv_cov = train_dat_c[train_dat_c.Subject.isin(test_cv.Subject)]
#    test_cv_cov = pd.merge(test_cv_cov, test_cv.Subject, how='inner', on='Subject')
#    test_cv_im = test_cv_im.reset_index(drop=True)
#    test_cv_cov = test_cv_cov.reset_index(drop=True)
#    test_cv_f = pd.concat([test_cv_cov, test_cv_im], axis=1)
#    test_cv_br = test_cv_f.drop(covs, axis=1)
#    print(test_cv_br.columns)
##
#    filename = 'va_%d_rep_%d.csv'%(fold_size,f)
#    va_cv=pd.read_csv(folder_path+filename)
#    # get the image data
#    va_cv_im = extract_features_from_cnn(cfg, 'va', f, '../deep_learning/BMI/results/cv/')
#    va_cv_im['Subject'] = va_cv['Subject']    
#    #va_cv_im = va_cv_im[va_cv.Family_ID.isin(train_dat_c.Family_ID)]
#    #va_cv_cov = train_dat_c[train_dat_c.Family_ID.isin(va_cv.Family_ID)]
#    # Merge the DataFrames based on the common column
#    va_cv_im = pd.merge(va_cv_im, train_dat_c.Subject, how='inner', on='Subject')
#    #train_cv_im = train_cv_im[train_cv.Family_ID.isin(train_dat_c.Family_ID)]
#    # Sort train_cv_cov based on the order of 'Family_ID' in train_cv
#    va_cv_cov = train_dat_c[train_dat_c.Subject.isin(va_cv.Subject)]
#    va_cv_cov = pd.merge(va_cv_cov, va_cv.Subject, how='inner', on='Subject') 
#    va_cv_im = va_cv_im.reset_index(drop=True)
#    va_cv_cov = va_cv_cov.reset_index(drop=True)
#    va_cv_f = pd.concat([va_cv_cov, va_cv_im], axis=1)
#    va_cv_br = va_cv_f.drop(covs, axis=1)
#    # Save datasets for future use
#    fold_size = train_cv.shape[0]
#    train_cv_f.to_csv("ssd_tabnet_cnn/cv/tr_%d_rep_%d_full.csv" %(fold_size, f), index=False)
#    test_cv_f.to_csv("ssd_tabnet_cnn/cv/te_%d_rep_%d_full.csv" %(fold_size, f), index=False)
#    va_cv_f.to_csv("ssd_tabnet_cnn/cv/va_%d_rep_%d_full.csv" %(fold_size, f), index=False)
#    train_cv_br.to_csv("ssd_tabnet_cnn/cv/tr_%d_rep_%d_brain.csv" %(fold_size, f), index=False)
#    test_cv_br.to_csv("ssd_tabnet_cnn/cv/te_%d_rep_%d_brain.csv" %(fold_size, f), index=False)
#    va_cv_br.to_csv("ssd_tabnet_cnn/cv/va_%d_rep_%d_brain.csv" %(fold_size, f), index=False)
#    train_cv_cov.to_csv("ssd_tabnet_cnn/cv/tr_%d_rep_%d_cov.csv" %(fold_size, f), index=False)
#    test_cv_cov.to_csv("ssd_tabnet_cnn/cv/te_%d_rep_%d_cov.csv" %(fold_size, f), index=False)
#    va_cv_cov.to_csv("ssd_tabnet_cnn/cv/va_%d_rep_%d_cov.csv" %(fold_size, f), index=False)
#    del train_cv_cov
#    del train_cv_br
#    del train_cv_f
#    del test_cv_cov
#    del test_cv_br
#    del test_cv_f
#    del va_cv_cov
#    del va_cv_br
#    del va_cv_f
#
## In[64]:
#
#
### now save the nested cv folds
## Define the folder path where data files are located
#folder_path = '../deep_learning/BMI/ssd/cv/'
#### Loop through the files in the folder
#fold_sizes = joblib.load(folder_path+"/fold_sizes.pkl")
## Loop through the files in the folder
#for f in range(15):
#    fold_size = fold_sizes[f]
#    # read in train_cv from CNN feature data
#    train_cv_full = pd.read_csv("ssd_tabnet_cnn/cv/tr_%d_rep_%d_full.csv" %(fold_size, f))
#    train_cv_cov = pd.read_csv("ssd_tabnet_cnn/cv/tr_%d_rep_%d_cov.csv" %(fold_size, f))
#    train_cv_brain = pd.read_csv("ssd_tabnet_cnn/cv/tr_%d_rep_%d_brain.csv" %(fold_size, f))
#    for k in range(3):
#        folder_path = '../deep_learning/BMI/ssd/cv/hyp_tune/fold_%d/' %f
#        #folder_path = '../deep_learning/BMI/ssd/cv/hyp_tune/fold_%d/' %f
#
#        for filename in os.listdir(folder_path):
#            if 'tr' in filename and 'rep_%d' %k in filename:
#                # read in train_cv for nested fold from CNN
#                train_nest=pd.read_csv(folder_path+filename)
#                # read in train_cv from
#                train_cv2_cov = train_cv_cov[train_cv_cov.Family_ID.isin(train_nest.Family_ID)]
#                train_cv2_br = train_cv_brain[train_cv_brain.Family_ID.isin(train_nest.Family_ID)]
#                train_cv2_f = train_cv_full[train_cv_full.Family_ID.isin(train_nest.Family_ID)]
#
#
#                # Define a regular expression pattern to match numbers
#                #pattern = r"\d+"
#                # Use re.search to find the first occurrence of the pattern
#                #match = re.search(pattern, filename)
#                #fold_size = int(match.group())
#                #print(fold_size)
#                #cfg = ut.Config(ssd=folder_path, predictor='smriPath', scorename='outcome', cuda_avl=True,
#                #                cr='reg', tss=fold_size, rep=k)
#                # get the image data
#                #train_cv_im = extract_features_from_cnn(cfg, 'tr', k, '../deep_learning/BMI/results/cv/hyp_tune/fold_%d/'%f)
#                #train_cv_cov = train_dat_c[train_dat_c.Family_ID.isin(train_cv.Family_ID)]
#                #train_cv_im = train_cv_im.reset_index(drop=True)
#                #train_cv_cov = train_cv_cov.reset_index(drop=True)
#                #train_cv_f = pd.concat([train_cv_cov, train_cv_im], axis=1)
#                #print(train_cv_f.columns)
#                #train_cv_br = train_cv_f.drop(covs, axis=1)
#
#            elif 'te' in filename and 'rep_%d' %k in filename:
#                test_nest=pd.read_csv(folder_path+filename)
#                test_cv2_cov = train_cv_cov[train_cv_cov.Family_ID.isin(test_nest.Family_ID)]
#                test_cv2_br = train_cv_brain[train_cv_brain.Family_ID.isin(test_nest.Family_ID)]
#                test_cv2_f = train_cv_full[train_cv_full.Family_ID.isin(test_nest.Family_ID)]
#
#                # Define a regular expression pattern to match numbers
#                #pattern = r"\d+"
#                # Use re.search to find the first occurrence of the pattern
#                #match = re.search(pattern, filename)
#                #fold_size = int(match.group())
#                #print(fold_size)
#                #cfg = ut.Config(ssd=folder_path, predictor='smriPath', scorename='outcome', cuda_avl=True,
#                #                cr='reg', tss=fold_size, rep=k)
#                # get the image data
#                #test_cv_im = extract_features_from_cnn(cfg, 'te', k, '../deep_learning/BMI/results/cv/hyp_tune/fold_%d/'%f)
#                #test_cv_cov = train_dat_c[train_dat_c.Family_ID.isin(test_cv.Family_ID)]
#                #test_cv_im = test_cv_im.reset_index(drop=True)
#                #test_cv_cov = test_cv_cov.reset_index(drop=True)
#                #test_cv_f = pd.concat([test_cv_cov, test_cv_im], axis=1)
#                #print(train_cv_f.columns)
#                #test_cv_br = test_cv_f.drop(covs, axis=1)
#
#            elif 'va' in filename and 'rep_%d' %k in filename:
#                va_nest=pd.read_csv(folder_path+filename)
#                va_cv2_cov = train_cv_cov[train_cv_cov.Family_ID.isin(va_nest.Family_ID)]
#                va_cv2_br = train_cv_brain[train_cv_brain.Family_ID.isin(va_nest.Family_ID)]
#                va_cv2_f = train_cv_full[train_cv_full.Family_ID.isin(va_nest.Family_ID)]
#                # Define a regular expression pattern to match numbers
#                #pattern = r"\d+"
#                # Use re.search to find the first occurrence of the pattern
#                #match = re.search(pattern, filename)
#                #fold_size = int(match.group())
#                #print(fold_size)
#                #cfg = ut.Config(ssd=folder_path, predictor='smriPath', scorename='outcome', cuda_avl=True,
#                #                cr='reg', tss=fold_size, rep=k)
#                # get the image data
#                #va_cv_im = extract_features_from_cnn(cfg, 'va', k, '../deep_learning/BMI/results/cv/hyp_tune/fold_%d/'%f)
#                #va_cv_cov = train_dat_c[train_dat_c.Family_ID.isin(va_cv.Family_ID)]
#                #va_cv_im = va_cv_im.reset_index(drop=True)
#                #va_cv_cov = va_cv_cov.reset_index(drop=True)
#                #va_cv_f = pd.concat([va_cv_cov, va_cv_im], axis=1)
#                #print(va_cv_f.columns)
#                #va_cv_br = va_cv_f.drop(covs, axis=1)
#        
#        # Save datasets for future use
#        tss = train_nest.shape[0]
#        print(tss)
#        fold_path_tab = "ssd_tabnet_cnn/cv/hyp_tune/fold_%d/" %f
#        # Make Model Directory
#        
#        try:
#            os.stat(fold_path_tab)
#        except:
#            os.makedirs(fold_path_tab)
#        print(train_cv2_cov.shape[0])
#        train_cv2_f.to_csv(fold_path_tab+"tr_%d_rep_%d_full.csv" %(tss, k), index=False)
#        test_cv2_f.to_csv(fold_path_tab+"te_%d_rep_%d_full.csv" %(tss, k), index=False)
#        va_cv2_f.to_csv(fold_path_tab+"va_%d_rep_%d_full.csv" %(tss, k), index=False)
#        train_cv2_br.to_csv(fold_path_tab+"tr_%d_rep_%d_brain.csv" %(tss, k), index=False)
#        test_cv2_br.to_csv(fold_path_tab+"te_%d_rep_%d_brain.csv" %(tss, k), index=False)
#        va_cv2_br.to_csv(fold_path_tab+"va_%d_rep_%d_brain.csv" %(tss, k), index=False)
#        train_cv2_cov.to_csv(fold_path_tab+"tr_%d_rep_%d_cov.csv" %(tss, k), index=False)
#        test_cv2_cov.to_csv(fold_path_tab+"te_%d_rep_%d_cov.csv" %(tss, k), index=False)
#        va_cv2_cov.to_csv(fold_path_tab+"va_%d_rep_%d_cov.csv" %(tss, k), index=False)
#        del train_cv2_cov
#        del train_cv2_br
#        del train_cv2_f
#        del test_cv2_cov
#        del test_cv2_br
#        del test_cv2_f
#        del va_cv2_cov
#        del va_cv2_br
#        del va_cv2_f
#
## In[ ]:
#
#
#
#
#
