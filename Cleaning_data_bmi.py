#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[220]:


import glob
import re
# read in image file names
#img_names = glob.glob("brain_files/*")


# In[221]:


#string = "C:/Users/alysh/OneDrive/Documents/PBCAR/Deep Learning HCP/brain_files/"

img_names = glob.glob("/project/6046119/sharing/hcp_gm_segs/final_gmv_files_1107/*")
#print(img_names)
#print(len(img_names))
#string = "C:/Users/alysh/OneDrive/Documents/PBCAR/Deep Learning HCP/brain_files/"
string = "/project/6046119/sharing/hcp_gm_segs/final_gmv_files_1107/"
# In[222]:

img_names = [name.rsplit('/')[-1] for name in img_names]

ID = []
for i in range(len(img_names)):
    ID.append(re.findall(r'\d+', img_names[i]))
ID = [int(item[1:]) for sublist in ID for item in sublist]
print(img_names)
print(ID)

img_names = [string + name for name in img_names]

#ID = []
#for i in range(len(img_names)):
#    ID.append(re.findall(r'\d+', img_names[i]))
#ID
# In[223]:


#import re


# In[224]:


#ID = []
#for i in range(len(img_names)):
#    ID.append(re.findall(r'\d+', img_names[i]))
#ID = [int(item[1:]) for sublist in ID for item in sublist]
#print(len(img_names))
#print(ID)

# In[225]:


import pandas as pd
rest_dat = pd.read_csv("/home/acoope05/Tabnet/Test_run/RESTRICTED_owensmax_2_8_2021_13_31_28.csv")
unrest_dat = pd.read_csv("/home/acoope05/Tabnet/Test_run/unrestricted_owensmax_2_2_2021_13_58_55.csv")


# In[226]:


import numpy as np
print(np.sum(rest_dat.Subject == unrest_dat.Subject))


# In[227]:


# remove inconsistencies 

## Checking for and removing inconsistencies

# Create list of delayed discounting variables
#drd_qc_vars = ['DDisc_SV_1mo_200', 'DDisc_SV_6mo_200', 'DDisc_SV_1yr_200',
#'DDisc_SV_3yr_200', 'DDisc_SV_5yr_200', 'DDisc_SV_10yr_200',
#'DDisc_SV_1mo_40K', 'DDisc_SV_6mo_40K', 'DDisc_SV_1yr_40K',
#'DDisc_SV_3yr_40K', 'DDisc_SV_5yr_40K', 'DDisc_SV_10yr_40K']

# Find inconsistencies
#unrest_dat['qc0'] = unrest_dat[drd_qc_vars[0]] < unrest_dat[drd_qc_vars[1]] # do we need this row?
#unrest_dat['qc1'] = unrest_dat[drd_qc_vars[1]] < unrest_dat[drd_qc_vars[2]]
#unrest_dat['qc2'] = unrest_dat[drd_qc_vars[2]] < unrest_dat[drd_qc_vars[3]]
#unrest_dat['qc3'] = unrest_dat[drd_qc_vars[3]] < unrest_dat[drd_qc_vars[4]]
#unrest_dat['qc4'] = unrest_dat[drd_qc_vars[4]] < unrest_dat[drd_qc_vars[5]]

#unrest_dat['qc5'] = unrest_dat[drd_qc_vars[6]] < unrest_dat[drd_qc_vars[7]]
#unrest_dat['qc6'] = unrest_dat[drd_qc_vars[7]] < unrest_dat[drd_qc_vars[8]]
#unrest_dat['qc7'] = unrest_dat[drd_qc_vars[8]] < unrest_dat[drd_qc_vars[9]]
#unrest_dat['qc8'] = unrest_dat[drd_qc_vars[9]] < unrest_dat[drd_qc_vars[10]]
#unrest_dat['qc9'] = unrest_dat[drd_qc_vars[10]] < unrest_dat[drd_qc_vars[11]] # do we need this row?

#qc_vars = ['qc0', 'qc1', 'qc2', 'qc3', 'qc4', 'qc5', 'qc6', 'qc7', 'qc8', 'qc9']

# Number of inconsistencies per person
#unrest_dat['inconsistencies'] = unrest_dat[qc_vars].sum(axis=1)
#print("Number of individuals removed: %d" %np.sum(unrest_dat['inconsistencies']>=4)) # 35 if we add first and last row, else 65

# Remove those with >= 4 inconsistencies
#rest_dat = rest_dat[unrest_dat['inconsistencies'] < 4]
#unrest_dat = unrest_dat[unrest_dat['inconsistencies'] < 4]
#ID  = [num for num in ID if num in list(set(ID) & set(unrest_dat.Subject))]
#print(len(ID))

# In[228]:


## Create outcome variable

# total cognitive score is outcome
rest_dat = rest_dat.assign(outcome=rest_dat[["BMI"]])


# In[230]:
print("The number of image files is: %d" % len(img_names))
print("The shape of the tabular restricted data is: %d" % rest_dat.shape[0])
print("The shape of the tabular unrestricted data is: %d" % unrest_dat.shape[0])
print(len(img_names))


collect_dat = pd.DataFrame({"Subject":ID, "smriPath":img_names})


# In[231]:

#print(collect_dat.shape)
full_data = collect_dat.merge(rest_dat[["Subject", "Family_ID", "outcome"]], how='inner', on='Subject')
print("the data size combined: %d" % full_data.shape[0])
# Drop those with missing values 
full_data = full_data.dropna()
print("the data size with missing subject, family_id, outcome removed: %d" % full_data.shape[0])

# In[232]:


from sklearn.model_selection import train_test_split
# split dataframe into tr, te, va 
data_id           = full_data["Family_ID"].unique()
train_id, test_id = train_test_split(data_id, test_size=0.20, random_state=8)
val_id, test_id  = train_test_split(test_id, test_size=0.50, random_state=8)

# Create train, test, and validation sets for full data
train_f = full_data.loc[full_data["Family_ID"].isin(train_id), :]
test_f  = full_data.loc[full_data["Family_ID"].isin(test_id), :]
val_f   = full_data.loc[full_data["Family_ID"].isin(val_id), :]

print("Final train: n=%d, final test: n=%d, final validation: n=%d" % (train_f.shape[0], test_f.shape[0], val_f.shape[0]))


# In[233]:


# save dataframe in ssd folder 
#tr_size = train_f.shape[0]
#file_name_tr = ("ssd/tr_%d_rep_0.csv" %(tr_size))
#file_name_te = ("ssd/te_%d_rep_0.csv" %(tr_size))
#file_name_va = ("ssd/va_%d_rep_0.csv" %(tr_size))
#train_f.to_csv(file_name_tr, index=False)
#test_f.to_csv(file_name_te, index=False)
#val_f.to_csv(file_name_va, index=False)

