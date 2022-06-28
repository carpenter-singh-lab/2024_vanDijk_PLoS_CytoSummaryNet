## Standard libraries
import os
import numpy as np
import time
import pandas as pd
import random

## tqdm for loading bars
from tqdm import tqdm

## PyTorch
import torch
import torch.utils.data as data
import torch.optim as optim

# Losses
from pytorch_metric_learning import losses, distances

# Custom libraries
from networks.SimpleMLPs import MLPsumV2
from dataloader_pickles import DataloaderTrainVX, DataloaderEvalV5
import utils

#%% Set random seed for reproducibility
manualSeed = 42
print("Random Seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
# Set device for GPU usage
torch.cuda.empty_cache()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#%% Set hyperparameters

lr = 1e-3  # learning rate
epochs = 25  # maximum number of epochs
nr_sets = 4 # number of sets you sample per well
BS = 18  # 54 for grouped // batch size as passed to the model (nr of wells that you sample per batch)
true_BS = int(nr_sets*BS) # true number of sets that the model will see per batch
print(f'True batch size is {true_BS}')
# nr of cells sampled from each well (no more than 1200 found in compound plates)
# May be good to look at the average number of cells per well for each dataset or tune as hyperparameter
initial_cells = 2000
nr_cells = (initial_cells, 900)  # INTEGER or TUPLE (median, std) for gaussian // (1100, 300)
input_dim = 1324
kFilters = 1/2 # times DIVISION of filters in model
latent_dim = 2048
output_dim = 1024
dropout = 0.0

weight_decay = 'AdamW default'

#%% Load all data

model_name = 'general_ckpt_simpleMLP_V1_sphered'

rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/Stain3'
metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv', index_col=False)
plateDirs = [x[0] for x in os.walk(rootDir)][1:]

plates = ['BR00115134_FS', 'BR00115125_FS', 'BR00115133highexp_FS'] # STAIN3

plateDirs = [x for x in plateDirs if any(substr in x for substr in plates)]

TrainLoaders = []
ValLoaders = []
for i, pDir in enumerate(plateDirs):
    C_metadata = utils.addDataPathsToMetadata(rootDir, metadata, pDir)
    df = utils.filterData(C_metadata, 'negcon', encode='pert_iname')
    TrainTotal, ValTotal = utils.train_val_split(df, 0.8)

    gTDF = TrainTotal.groupby('Metadata_labels')
    trainset = DataloaderTrainVX(TrainTotal, nr_cells=initial_cells, nr_sets=nr_sets, groupDF=gTDF, preprocess='sphere')
    TrainLoaders.append(data.DataLoader(trainset, batch_size=BS, shuffle=True, collate_fn=utils.my_collate,
                                        drop_last=False, pin_memory=False, num_workers=0))

    valset = DataloaderEvalV5(ValTotal)
    ValLoaders.append(data.DataLoader(valset, batch_size=1, shuffle=False,
                                   drop_last=False, pin_memory=False, num_workers=0))


print(f'\nLoading {len(TrainLoaders)} plates. Did you check the training loop?')
if input('Continue? [y/n]') == 'y':
    pass
else:
    raise Warning('Stopping run')

# %% Setup models
model = MLPsumV2(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim,
                 k=kFilters, dropout=0, cell_layers=1,
                 proj_layers=2, reduction='sum')
if torch.cuda.is_available():
    model.cuda()
# %% Setup optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr)
loss_func = losses.SupConLoss(distance=distances.CosineSimilarity())

# %% Start training
print(utils.now() + "Start training")
best_val = np.inf

for e in range(epochs):
    time.sleep(0.5)
    model.train()
    tr_loss = 0.0

    print("Training epoch")
    for idx, data in enumerate(tqdm(zip(TrainLoaders[0], TrainLoaders[1], TrainLoaders[2]), total=len(TrainLoaders[0]))):
        points, labels = [d[0] for d in data], [d[1] for d in data]
        points, labels = [x.to(device) for x in points], [y.to(device) for y in labels]

        temp_losses = []
        for P, L in zip(points, labels):
            # Retrieve feature embeddings
            feats, _ = model(P)
            # Calculate loss
            temp_losses.append(loss_func(feats, L))

        tr_loss_tmp = sum(temp_losses) / len(TrainLoaders)
        # add the loss to running variable
        tr_loss += tr_loss_tmp.item()

        # Adam
        tr_loss_tmp.backward()
        optimizer.step()
        optimizer.zero_grad()

        # RESET NR_CELLS
        if isinstance(nr_cells, tuple):
            CELLS = int(np.random.normal(nr_cells[0], nr_cells[1], 1))
            while CELLS < 100:
                CELLS = int(np.random.normal(nr_cells[0], nr_cells[1], 1))
            for z in range(len(TrainLoaders)):
                TrainLoaders[z].dataset.nr_cells = CELLS

    tr_loss /= (idx+1)

    # Validation
    model.eval()
    val_loss = 0.0
    time.sleep(0.5)
    print('Validation epoch')
    time.sleep(0.5)

    temp_losses = []
    with torch.no_grad():
        for dataloader_idx in tqdm(range(len(ValLoaders))):
            MLP_profiles = torch.tensor([], dtype=torch.float32)
            MLP_labels = torch.tensor([], dtype=torch.int16)
            for (points, labels) in ValLoaders[dataloader_idx]:
                feats, _ = model(points)
                # Append everything to dataframes
                MLP_profiles = torch.cat([MLP_profiles, feats])
                MLP_labels = torch.cat([MLP_labels, labels])

            temp_losses.append(loss_func(MLP_profiles, MLP_labels))

        val_loss_tmp = sum(temp_losses) / len(ValLoaders)
        # add the loss to running variable
        val_loss += val_loss_tmp.item()

    print(utils.now() + f"Epoch {e}. Training loss: {tr_loss}. Validation loss: {val_loss}.")

    print('Creating model checkpoint...')
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tr_loss': tr_loss,
        'val_loss': val_loss,
    }, os.path.join('/Users/rdijk/Documents/ProjectFA/InterpretabilityAnalysis/LearningCovarianceMatrices/models',
                    model_name))

print(utils.now() + 'Finished training')


