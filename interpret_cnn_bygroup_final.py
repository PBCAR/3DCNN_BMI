import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#from torch.autograd import Variable
from torch.autograd import Variable
import utils_tune as ut
import nibabel as nib
import pandas as pd
import os
from scipy import ndimage
from dataclasses import dataclass


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

#--- load in net
#net = ut.AlexNet3D_Dropout_Regression(num_classes=1)
#model = torch.nn.DataParallel(net)
#net = ut.load_net_weights2(model, 'results/AlexNet3D_Dropout_Regression_scorename_outcome_tss_886_rep_0_bs_32_lr_0.00017144423766524508_espat_40/model_state_dict.pt')
#net.eval()
#net.zero_grad()

# find fold with lowest MSE
path = ("results/cv/")
cv_summary = pd.read_csv(path+"cv_summary.csv")
best_fold = cv_summary['mse_te'].idxmin()

# Retrieve the directory from the best fold row
best_directory = cv_summary.loc[best_fold, 'Directory']
model_state_path = f"results/cv/{best_directory}/model_state_dict.pt"
print(model_state_path)

# read in MNI template
#brain_img = nib.load("mni152.nii.gz")


# Load in the model
#--- load in net
net = ut.AlexNet3D_Dropout_Regression(num_classes=1)
model = torch.nn.DataParallel(net)
net = ut.load_net_weights2(model, model_state_path)
net.eval()
net.zero_grad()

# re-setup network to add hook functions
class my_net(nn.Module):
    def __init__(self, net):
        super(my_net, self).__init__()
        
        # get the pretrained network
        self.net = net
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.net.module.features[:17]
        
        # get the max pool of the features stem
        self.max_pool = self.net.module.features[17]
        
        # get the regressor
        self.regressor = self.net.module.regressor
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.regressor(x)
        print(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


# define my new network
bmi_net = my_net(net)
bmi_net.eval()


#--- function to read in data with subject

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

    return X,y,z


#--- ensure model is on GPU
def check_model_device(model):
    return next(model.parameters()).device

#--- function to generate heatmap



def generate_heatmap(X, brain_img, bmi_net):
    X = torch.unsqueeze(torch.tensor(X), 1)
    #print(X.shape)
    X = Variable(X.cuda())

    # Check if model is on GPU, if not, move it
    model_device = check_model_device(bmi_net)
    if model_device != torch.device('cuda'):
        bmi_net = bmi_net.cuda()
        print("Model moved to GPU")
    else:
        print("Model is already on GPU")



    bmi_net.eval()
    bmi_net.zero_grad()
    
    # get prediction for X
    pred=bmi_net(X)

    # get the gradient
    pred.backward()
    gradients = bmi_net.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0,2,3,4])
    activations = bmi_net.get_activations(X).detach()
    for i in range(128):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # up sample and re-align heatmap
    # Calculate zoom factors to match brain image shape
    zoom_factors = np.array(brain_img.shape) / np.array(heatmap.shape)

    # Resample the heatmap to match brain image dimensions
    resampled_heatmap_data = ndimage.zoom(heatmap.cpu().numpy(), zoom_factors, order=3)

    # Create a new NIfTI image with the resampled heatmap data, using the brain image affine
    resampled_heatmap_img = nib.Nifti1Image(resampled_heatmap_data, brain_img.affine)

    # Calculate the center of the heatmap and MNI template using their shapes and affine matrices
    def get_image_center(shape, affine):
        center_voxel = np.array(shape) / 2.0  # Get the voxel space center
        center_world = affine.dot(np.append(center_voxel, 1))[:3]  # Convert to world space (x, y, z)
        return center_world

    heatmap_center = get_image_center(resampled_heatmap_img.shape, resampled_heatmap_img.affine)
    mni_center = get_image_center(brain_img.shape, brain_img.affine)

    print(f"Heatmap center (world coordinates): {heatmap_center}")
    print(f"MNI Template center (world coordinates): {mni_center}")

    # Calculate the shift needed to align the centers
    shift = mni_center - heatmap_center
    print(f"Shift required to align centers: {shift}")

    # Modify the heatmap's affine matrix to apply the shift
    aligned_affine = resampled_heatmap_img.affine.copy()
    aligned_affine[:3, 3] += shift  # Apply the calculated shift to the translation part of the affine matrix

    shifted_heatmap_img = nib.Nifti1Image(resampled_heatmap_img.get_fdata(), aligned_affine)

    shifted_heatmap_data = shifted_heatmap_img.get_fdata()
    # normalize the heatmap
    norm_heatmap = (shifted_heatmap_data-np.mean(shifted_heatmap_data))/(np.std(shifted_heatmap_data))

    return norm_heatmap


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
#dset = MRIDataset(cfg, 'combined')


df = readFrames(cfg.ssd, 'combined', cfg.tss, cfg.rep)

#print(dset.__len__())
#print(dset.__getitem__(0))
#n = dset.__len__()
n = df.shape[0]


# Initialize an empty list for the new concatenated data
new_X = []
new_Y = []
new_Z = []
brains = []

# Loop through the observation
# Loop through the observations
for i in range(n):
    #X, Y, Z = dset.__getitem__(i)


    Z = df[cfg.sub_id].iloc[i]
    Z = np.array(Z)
    # Read image
    fN = df[cfg.predictor].iloc[i]
    brain_img = nib.load(fN)
    X = np.float32(nib.load(fN).get_fdata())
    X = (X - X.min()) / (X.max() - X.min())
    X = np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))
    # Read label
    Y = df[cfg.scorename].iloc[i]
    Y = np.array(np.float32(Y))

    # Convert X to NumPy array, add a new channel axis, and append it to the new_X list
    new_observation = np.expand_dims(X, axis=0)
    brains.append(brain_img)
    new_X.append(new_observation)
    new_Y.append(Y)
    new_Z.append(Z)

# Stack the list of 3D arrays along the first axis to get the final array
final_X = np.vstack(new_X)
final_Y = np.vstack(new_Y)
final_Z = np.vstack(new_Z)
#print(final_X.shape)
#print(final_Y.shape)

#final_YZ = pd.DataFrame((final_Y, final_Z))

# Convert to DataFrame
final_YZ = pd.DataFrame({'Observed1': final_Y.squeeze(), 'Subject': final_Z.squeeze()})

# Create a list of tuples (label, input) for sorting
#data_tuples = list(zip(labels, inputs))

# Sort the data based on labels in descending order
#sorted_data = sorted(data_tuples, key=lambda x: x[0], reverse=True)

# Unzip the sorted data to get sorted inputs and labels
#sorted_labels, sorted_inputs = zip(*sorted_data)


#sorted_indices = np.argsort(final_Y[:, 0])[::-1]

# Use the sorted indices to reorder final_X and final_Y
#sorted_final_X = final_X[sorted_indices]
#sorted_final_Y = final_Y[sorted_indices]

#print(sorted_final_Y)

# read in predictions
predictions = pd.read_csv("results/predictions_features_alt.csv")

# merge predictions and observed based on subject
df_merged = pd.merge(predictions, final_YZ, on="Subject")
#print(final_YZ.head())
#print(df_merged.head())

# Example indices
conditions = [
    ('h_pred_o', (df_merged['Observed'] < 25) & (df_merged['Observed'] >= 18.5) & (df_merged['Prediction'] >= 25)),
    ('h_pred_h',  (df_merged['Observed'] < 25) & (df_merged['Observed'] >= 18.5) & (df_merged['Prediction'] < 25) & (df_merged['Prediction'] >= 18.5)),
    ('h_pred_u', (df_merged['Observed'] < 25) & (df_merged['Observed'] > 18.5) & (df_merged['Prediction'] <= 18.5)),
    ('o_pred_h', (df_merged['Observed'] >= 25) & (df_merged['Prediction'] < 25) & (df_merged['Prediction'] >= 18.5)),
    ('o_pred_o', (df_merged['Observed'] >= 25) & (df_merged['Prediction'] >= 25))
]

for condition_name, condition_mask in conditions:
    # Get indices based on condition
    condition_indices = np.where(condition_mask)[0]
    # Print the indices found
    print(f"Indices for {condition_name}:")
    print(condition_indices)    
    selected_heatmaps = []
    for idx in condition_indices:
        brain_img = brains[idx]
        #print(final_Z[idx])
        X_observation = final_X[idx]  # Assuming final_X is defined somewhere
        heatmap = generate_heatmap(X_observation, brain_img, bmi_net)  # Assuming bmi_net is your model
        selected_heatmaps.append(heatmap)

    # find mean heatmap across subjects
    mean_heatmap = np.mean(selected_heatmaps, axis=0)

    # Calculate the 75th percentile
    #percentile_75 = torch.kthvalue(mean_heatmap.flatten(), int(0.975 * mean_heatmap.numel())).values
    #percentile_75 = np.percentile(mean_heatmap.flatten(), q = 98)
    #print(percentile_75)

    # Set values below the 2 sds from mean to zero
    mean_heatmap[abs(mean_heatmap) < 2] = 0
    print(mean_heatmap.shape)
    #print(np.sum(mean_heatmap))
    #new_min = np.min(mean_heatmap[mean_heatmap>0])
    #zoom_factor_cam = (X_observation.shape[1]/mean_heatmap.shape[0], X_observation.shape[2]/mean_heatmap.shape[1], X_observation.shape[3]/mean_heatmap.shape[2])

    #cam = ndimage.zoom(mean_heatmap, zoom_factor_cam)
    #cam[cam<new_min]=0
    #cam = np.clip(cam, 0, 1)
    #print(cam.shape)
    #percentile_97 = np.percentile(abs(mean_heatmap), 97.5)
    #print(mean_heatmap)
    # Set values below the 75th percentile to zero
    #mean_heatmap[abs(mean_heatmap) < percentile_97] = 0

    # Save the heatmap as NIfTI file
    final_img = nib.Nifti1Image(mean_heatmap, affine=brain_img.affine)
    file_path = os.path.join('results', f"{condition_name}_heatmap_final_cubic.nii.gz")
    nib.save(final_img, file_path)
