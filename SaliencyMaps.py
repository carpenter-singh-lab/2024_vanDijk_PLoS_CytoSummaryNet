## Standard libraries
import os
from tqdm import tqdm
import pandas as pd

## Seeds
import random
import numpy as np

## PyTorch
import torch
import torch.utils.data as data

# Custom libraries
from networks.SimpleMLPs import MLP
from dataloader_pickles import DataloaderEvalV5
import utils

NUM_WORKERS = 0
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

# Set random seed for reproducibility
manualSeed = 42
# manualSeed = random.randint(1,10000) # use if you want new results
print("Random Seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# %% Load model
save_name_extension = 'general_ckpt_simpleMLP_V1'
model_name = save_name_extension
print('Loading:', model_name)

input_dim = 1324
kFilters = 4
latent_dim = 1028
output_dim = 512
model = MLP(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim, k=kFilters)
if torch.cuda.is_available():
    model.cuda()

save_features_to_csv = True
evaluate_point_distributions = False
path = r'wandb/latest-run/files'

models = os.listdir(path)
fullpath = os.path.join(path, model_name)
if 'ckpt' in model_name:
    model.load_state_dict(torch.load(fullpath)['model_state_dict'])
else:
    model.load_state_dict(torch.load(fullpath))
model.eval()
# %% Load all data
plateNR = 0
rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/Stain2'
metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv', index_col=False)
plateDirs = [x[0] for x in os.walk(rootDir)][1:]
plateDirs = [plateDirs[plateNR]] # EVALUATE ONLY SINGLE PLATE
platestring = plateDirs[0].split('_')[-2]
print('Calculating results for: ' + platestring)
metadata = utils.addDataPathsToMetadata(rootDir, metadata, plateDirs)

# Filter the data and create numerical labels
df_prep = utils.filterData(metadata, 'negcon', encode='pert_iname', sort=False)
# Add all data to one DF
Total, _ = utils.train_val_split(df_prep, 1.0, sort=False)

valset = DataloaderEvalV5(Total, feature_selection=False)
loader = data.DataLoader(valset, batch_size=1, shuffle=False,
                         drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)

#%%
print('Calculating Salient Features')
(points, labels) = next(iter(loader))
print(df_prep[['pert_iname', 'well_position']][df_prep.Metadata_labels == labels.item()])
points = points.to(device)
points.requires_grad_()
features, _ = model(points)

features_max = torch.sum(features) # SUM because then we can look at which features influence the entire output profile the most
features_max.backward()

# Retrieve the saliency map and also pick the maximum value from channels on each pixel.
# In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
saliency, _ = torch.max(points.grad.data.abs(), dim=1)

# Return top X features
top50_idx = np.argpartition(saliency.numpy(), -20)[0, -20:]

# Get Feature Names
df = pd.read_csv('/Users/rdijk/Documents/Data/RawData/CommonFeatureNames.csv', index_col=False)

newDF = pd.DataFrame()
newDF['FeatureNames'] = df.iloc[top50_idx]
newDF['Saliency'] = saliency[0, top50_idx]
print(newDF.sort_values(by='Saliency', ascending=False).to_string(index=False))
