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
from pytorch_metric_learning import losses, distances

# Custom libraries
import wandb
from networks.SimpleMLPs import MLPsumV2
from src.dataloader_pickles import DataloaderTrainV7, DataloaderEvalV5
import utils

NUM_WORKERS = 0

# Set random seed for reproducibility
manualSeed = 42
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

# %% Set hyperparameters
hyperparameter_defaults = dict(
    lr=1e-3,  # learning rate
    epochs=30,  # maximum number of epochs
    nr_sets=4,  # number of sets you sample per well
    BS=18,  # batch size
    initial_cells=2000,
    cell_variance=900,
    kFilters=0.5,  # times DIVISION of filters in model
    input_dim=1324,
    latent_dim=2048,
    output_dim=1024,
    cell_layers=1,
    proj_layers=2,
    reduction='sum',
    weight_decay='AdamW default',
    dropout=0.0
)
wandb.init(project="FeatureAggregation", tags=['Sweep'], config=hyperparameter_defaults)
wandb.define_metric("Val loss", summary="min")
wandb.define_metric("Val mAP", summary="max")
config = wandb.config
config['nr_cells'] = (config['initial_cells'], config['cell_variance']),  # INTEGER or TUPLE (median, std) for gaussian // (1100, 300)

# %% Load all data
rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/Stain3'
metadata = pd.read_csv('/inputs/cpg0001_metadata/JUMP-MOA_compound_platemap_with_metadata.csv', index_col=False)
plateDirs = [x[0] for x in os.walk(rootDir)][1:]

plates = ['BR00115134_FS', 'BR00115125_FS', 'BR00115133highexp_FS']

plateDirs = [x for x in plateDirs if any(substr in x for substr in plates)]

TrainLoaders = []
ValLoaders = []
for i, pDir in enumerate(plateDirs):
    C_metadata = utils.addDataPathsToMetadata(rootDir, metadata, pDir)
    df = utils.filterData(C_metadata, 'negcon', encode='pert_iname')
    TrainTotal, _ = utils.train_val_split(df, 0.8)
    ValTotal, _ = utils.train_val_split(df, 1.0)

    gTDF = TrainTotal.groupby('Metadata_labels')
    trainset = DataloaderTrainV7(TrainTotal, nr_cells=config['initial_cells'], nr_sets=config['nr_sets'], groupDF=gTDF)
    TrainLoaders.append(data.DataLoader(trainset, batch_size=config['BS'], shuffle=True, collate_fn=utils.my_collate,
                                        drop_last=False, pin_memory=False, num_workers=NUM_WORKERS))
    valset = DataloaderEvalV5(ValTotal)
    ValLoaders.append(data.DataLoader(valset, batch_size=1, shuffle=False,
                                      drop_last=False, pin_memory=False, num_workers=NUM_WORKERS))


# %% Setup models
model = MLPsumV2(input_dim=config['input_dim'], latent_dim=config['latent_dim'], output_dim=config['output_dim'],
                 k=config['kFilters'], dropout=config['dropout'], cell_layers=config['cell_layers'],
                 proj_layers=config['proj_layers'], reduction=config['reduction'])
if torch.cuda.is_available():
    model.cuda()
# %% Setup optimizer
optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
loss_func = losses.SupConLoss(distance=distances.CosineSimilarity())

wandb.watch(model, loss_func, log='all', log_freq=10)

# %% Start training
best_val = 0

for e in tqdm(range(config['epochs'])):
    model.train()
    tr_loss = 0.0
    for idx, data in enumerate(zip(TrainLoaders[0], TrainLoaders[1], TrainLoaders[2])):
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
        if isinstance(config['nr_cells'], tuple):
            CELLS = int(np.random.normal(config['nr_cells'][0], config['nr_cells'][1], 1))
            while CELLS < 100 or CELLS % 2 != 0:
                CELLS = int(np.random.normal(config['nr_cells'][0], config['nr_cells'][1], 1))
            for z in range(len(TrainLoaders)):
                TrainLoaders[z].dataset.nr_cells = CELLS

    tr_loss /= (idx + 1)
    wandb.log({"Train Loss": tr_loss}, step=e)  # send data to wandb.ai

    # Validation
    model.eval()

    temp_losses = []
    temp_mAPs = []
    with torch.no_grad():
        for dataloader_idx in range(len(ValLoaders)):
            MLP_profiles = torch.tensor([], dtype=torch.float32)
            MLP_labels = torch.tensor([], dtype=torch.int16)
            for (points, labels) in ValLoaders[dataloader_idx]:
                feats, _ = model(points)
                # Append everything to dataframes
                MLP_profiles = torch.cat([MLP_profiles, feats])
                MLP_labels = torch.cat([MLP_labels, labels])

            temp_losses.append(loss_func(MLP_profiles, MLP_labels))

            # Calculate mAP
            MLP_profiles = pd.concat(
                [pd.DataFrame(MLP_profiles.detach().numpy()), pd.Series(MLP_labels.detach().numpy())], axis=1)
            MLP_profiles.columns = [f"f{x}" for x in range(MLP_profiles.shape[1] - 1)] + ['Metadata_label']
            AP = utils.CalculateMAP(MLP_profiles, 'cosine_similarity',
                                    groupby='Metadata_label', percent_matching=False)

            temp_mAPs.append(AP.AP.iloc[288:].mean())

        val_loss = sum(temp_losses) / len(temp_losses)
        val_loss = val_loss.item()
        val_mAP = sum(temp_mAPs) / len(temp_mAPs)

    if val_mAP > best_val:
        best_val = val_mAP

    wandb.log({"Val loss": val_loss, "Val mAP": val_mAP, "best_val_mAP": best_val}, step=e)  # send data to wandb.ai


