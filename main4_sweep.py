## Standard libraries
import os
import random
import numpy as np
import pandas as pd

## tqdm for loading bars
from tqdm import tqdm

## PyTorch
import torch
import torch.utils.data as data
import torch.optim as optim

# Losses
from pytorch_metric_learning import regularizers, losses, distances

# Custom libraries
import wandb
from networks.SimpleMLPs import MLP
from dataloader_pickles import DataloaderTrainV4, DataloaderTrainV5
import utils

NUM_WORKERS = 0

# Set random seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# %% Set hyperparameters
hyperparameter_defaults = dict(
    lr=1e-3,  # learning rate
    epochs=20,  # maximum number of epochs
    BS=16,  # batch size
    nr_cells=400,  # nr of cells sampled from each well (no more than 1200 found in compound plates)
    kFilters=1,  # times DIVISION of filters in model
    latent_dim=256,
    output_dim=128,
    nr_sets=4
)
wandb.init(project="FeatureAggregation", tags=['Sweep'], config=hyperparameter_defaults)  # 'dryrun'
config = wandb.config

# %% Load all data
rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/CPJUMP1'

# Set paths for all training/validation files
plateDirTrain1 = 'DataLoader_BR00117010_unfiltered'
tdir1 = os.path.join(rootDir, plateDirTrain1)
plateDirTrain2 = 'DataLoader_BR00117011_unfiltered'
tdir2 = os.path.join(rootDir, plateDirTrain2)
plateDirVal1 = 'DataLoader_BR00117012_unfiltered'
vdir1 = os.path.join(rootDir, plateDirVal1)
plateDirVal2 = 'DataLoader_BR00117013_unfiltered'
vdir2 = os.path.join(rootDir, plateDirVal2)

# Load csv for pair formation
metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/CPJUMP1_compounds/JUMP_target_compound_metadata_wells.csv', index_col=False)

# Load all absolute paths to the pickle files for training
filenamesTrain1 = [os.path.join(tdir1, file) for file in os.listdir(tdir1)]
filenamesTrain2 = [os.path.join(tdir2, file) for file in os.listdir(tdir2)]
filenamesTrain1.sort()
filenamesTrain2.sort()
# and for validation
filenamesVal1 = [os.path.join(vdir1, file) for file in os.listdir(vdir1)]
filenamesVal1.sort()
filenamesVal2 = [os.path.join(vdir2, file) for file in os.listdir(vdir2)]
filenamesVal2.sort()

# Create preprocessing dataframe for both validation and training
metadata['plate1'] = filenamesTrain1
metadata['plate2'] = filenamesTrain2
metadata['plate3'] = filenamesVal1
metadata['plate4'] = filenamesVal2

# Filter the data and create numerical labels
df = utils.filterData(metadata, 'negcon', encode='pert_iname')
# Split the data into training and validation set
TrainTotal, ValTotal = utils.train_val_split(df, 0.8)

trainset = DataloaderTrainV5(TrainTotal, nr_cells=config['nr_cells'], nr_sets=config['nr_sets'])
valset = DataloaderTrainV5(ValTotal, nr_cells=config['nr_cells'], nr_sets=config['nr_sets'])

train_loader = data.DataLoader(trainset, batch_size=config['BS'], shuffle=True, collate_fn=utils.my_collate,
                               drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)
val_loader = data.DataLoader(valset, batch_size=config['BS'], shuffle=True, collate_fn=utils.my_collate,
                             drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)

# %% Setup models
model = MLP(input_dim=1938, latent_dim=config['latent_dim'], output_dim=config['output_dim'],
            k=config['kFilters'])
# %% Setup optimizer
optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
loss_func = losses.SupConLoss(distance=distances.SNRDistance())

wandb.watch(model, loss_func, log='all', log_freq=10)

# %% Start training
best_val = np.inf

for e in range(config['epochs']):
    model.train()
    tr_loss = 0.0
    for idx, (points, labels) in enumerate(tqdm(train_loader)):
         # Send to device
        points, labels = points.to(device), labels.to(device)

        # Retrieve feature embeddings
        feats, _ = model(points)
        # Calculate loss
        tr_loss_tmp = loss_func(feats, labels)

        # add the loss to running variable
        tr_loss += tr_loss_tmp.item()

        # Adam
        tr_loss_tmp.backward()
        optimizer.step()
        optimizer.zero_grad()

    tr_loss /= (idx + 1)
    wandb.log({"Train Loss": tr_loss}, step=e)  # send data to wandb.ai

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (points, labels) in enumerate(tqdm(val_loader)):
            points, labels = points.to(device), labels.to(device)

            # Retrieve feature embeddings
            feats, _ = model(points)
            # Calculate loss
            val_loss_tmp = loss_func(feats, labels)

            # add the loss to running variable
            val_loss += val_loss_tmp.item()

    val_loss /= (i + 1)
    wandb.log({"Val loss": val_loss}, step=e)  # send data to wandb.ai

