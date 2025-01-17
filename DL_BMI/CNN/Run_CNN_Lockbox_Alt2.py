import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import utils_tune as ut
import nibabel as nib
import pandas as pd
from scipy import ndimage
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, explained_variance_score, mean_squared_error, r2_score
from scipy.stats import pearsonr

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



# find fold with lowest MSE
path = ("results/cv/")
cv_summary = pd.read_csv(path+"cv_summary.csv")
best_fold = cv_summary['mse_te'].idxmin()

# Retrieve the directory from the best fold row
best_directory = cv_summary.loc[best_fold, 'Directory']
model_state_path = f"results/cv/{best_directory}/model_state_dict.pt"

# Load in the model
#--- load in net
net = ut.AlexNet3D_Dropout_Regression(num_classes=1)
model = torch.nn.DataParallel(net)
net = ut.load_net_weights2(model, model_state_path)
net.eval()
net.zero_grad()

# set up configurations
@dataclass
class Config:
    nc: int = 10
    bs: int = 32
    lr: float = 0.001
    es: int = 1
    es_va: int = 1
    es_pat: int = 40
    ml: str = './temp/'
    mt: str = 'AlexNet3D_Dropout_Regression'
    ssd: str = '../SampleSplits/'
    predictor: str = 'smriPath'
    scorename: str = 'age'
    sub_id: str = 'Subject'
    cuda_avl: bool = True
    nw: int = 8
    cr: str = 'clx' # what is cr - classification (clx) or regression (reg)?
    tss: int = 100
    rep: int = 0

# define functions for reading in data
class MRIDataset(Dataset):

    def __init__(self, cfg, mode): # cfg from Config?
        self.df = readFrames(cfg.ssd, mode, cfg.tss, cfg.rep)
        self.scorename = cfg.scorename
        self.predictor = cfg.predictor
        self.sub_id    = cfg.sub_id
        self.cr = cfg.cr

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X, y, z = read_X_y_5D_idx(
            self.df, idx, self.predictor, self.scorename, self.sub_id, self.cr)
        return [X, y, z]


def readFrames(ssd, mode, tss, rep):

    # Read Data Frame
    df = pd.read_csv(ssd + mode + '_' + str(tss) +
                     '_rep_' + str(rep) + '.csv')

    print('Mode ' + mode + ' :' + 'Size : ' +
          str(df.shape) + ' : DataFrames Read ...')

    return df

# read in dataset as a 5D tensor with the labeled score (scorename) for the iterated id
def read_X_y_5D_idx(df, idx, predictor, scorename, sub_id, cr):

    X, y, z = [], [], []
    try:
        # read id
        z = df[sub_id].iloc[idx]
        z = np.array(z)

        # Read image
        fN = df[predictor].iloc[idx]
        X = np.float32(nib.load(fN).get_fdata())
        X = (X - X.min()) / (X.max() - X.min())
        X = np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))

        # Read label
        y = df[scorename].iloc[idx]
        if scorename == 'label':
            y -= 1

        if cr == 'reg':
            y = np.array(np.float32(y))
        elif cr == 'clx':
            y = np.array(y)
    except FileNotFoundError as e:
        print(f"Error: {e}. Skipping index {idx} and moving to the next one.")

   # Read ID
   #z = df[sub_id].iloc[idx]
   #z = np.array(z)
   

    return X, y, z





#--- function to obtain predictions



def get_preds(X, bmi_net):
    with torch.no_grad():
        X = torch.unsqueeze(torch.tensor(X), 1)
        #print(X.shape)
        X = Variable(X.cuda())
        # get prediction for X
        pred=bmi_net(X)
        pred_value = float(pred[0].item())
        # release GPU memory if using CUDA
        torch.cuda.empty_cache()
        print(pred_value)
        return pred_value


#-- read in test set

# define cfg

cfg =    Config(nc=1, 
                bs=1,
                ssd='ssd_recovered/ssd/', 
                predictor='smriPath', 
                scorename='outcome',
                sub_id='Subject', 
                cr='reg', 
                tss=886, 
                rep=0)
# fit dataset
dset = MRIDataset(cfg, 'combined')
print(dset.__len__())
#print(dset.__getitem__(0))
n = dset.__len__()

# Initialize an empty list for the new concatenated data
new_X = []
new_Y = []
new_z = []



# Loop through the observations and create list of all X, Y, z (Subject_ID)
for i in range(n):
    X, Y, z = dset.__getitem__(i)
    if Y:
         new_observation = np.expand_dims(X, axis=0)
         new_X.append(new_observation)
         #print(new_observation)
         new_Y.append(Y)
         new_z.append(z)
    else:
         print(f"Empty result for index {i}. Skipping this observation.")
# Stack the list of 3D arrays along the first axis to get the final array
final_X = np.vstack(new_X)
final_Y = np.vstack(new_Y)
final_z = np.vstack(new_z)
print(final_X.shape)
print(final_Y.shape)
print(final_z.shape)

#print(sorted_final_Y)

predictions = []

for idx in range(len(final_Y)):
    X_observation = final_X[idx]
    #id            = final_z[idx]
    pred = get_preds(X_observation, net)

    predictions.append(pred)


# Create a DataFrame with columns "Prediction" and "Subject"
df = pd.DataFrame({"Prediction": predictions, "Observed": final_Y.flatten(), "Subject": final_z.flatten()})
df.to_csv("results/predictions_alt.csv", index=False)

test_mae, test_ev, test_mse, test_r2, test_r, test_p = evaluate_reg(df.Observed, df.Prediction)

metrics = {
    "test_mae": [test_mae],
    "test_ev": [test_ev],
    "test_mse": [test_mse],
    "test_r2": [test_r2],
    "test_r": [test_r],
    "test_p": [test_p]                                                                                                               }

# Convert the dictionary to a DataFrame
metrics_df = pd.DataFrame(metrics)

# Save the DataFrame to a CSV file
metrics_df.to_csv("results/evaluation_metrics_lockbox_bestfold.csv", index=False)


