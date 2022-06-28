## Standard libraries
import os
import random
import numpy as np
import time
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
from networks.SimpleMLPs import MLP, MLPmean, MLPsum
from dataloader_pickles import DataloaderTrainV6
import utils

#%%
run = wandb.init(project="FeatureAggregation", mode='online', tags=['Generalized Models'])  # 'dryrun'
#%%
NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 0

# Set random seed for reproducibility
manualSeed = 42
# manualSeed = random.randint(1,10000) # use if you want new results
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
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

# %% Set hyperparameters
save_name_extension = 'simpleMLP_V1'  # extension of the saved model, specify architecture used

model_name = 'model_' + save_name_extension
print(model_name)

GROUP_WELLS = True

lr = 1e-3  # learning rate
epochs = 50  # maximum number of epochs
nr_sets = 4 # number of sets you sample per well
BS = 18  # 54 for grouped // batch size as passed to the model (nr of wells that you sample per batch)
true_BS = int(nr_sets*BS) # true number of sets that the model will see per batch
print(f'True batch size is {true_BS}')
# nr of cells sampled from each well (no more than 1200 found in compound plates)
# May be good to look at the average number of cells per well for each dataset or tune as hyperparameter
initial_cells = 1500
nr_cells = (initial_cells, 300)  # INTEGER or TUPLE (median, std) for gaussian // (1100, 300)
input_dim = 1324
kFilters = 1  # times DIVISION of filters in model
latent_dim = 1024  # test with 128
output_dim = 512

weight_decay = 'AdamW default'

#%% Load all Stain2 data
rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/Stain2'
metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv', index_col=False)
plateDirs = [x[0] for x in os.walk(rootDir)][1:]

plates = ['BR00113819', 'BR00113821']

plateDirs = [x for x in plateDirs if any(substr in x for substr in plates)]

TrainLoaders = []
ValLoaders = []
for i, pDir in enumerate(plateDirs):
    C_metadata = utils.addDataPathsToMetadata(rootDir, metadata, pDir)
    df = utils.filterData(C_metadata, 'negcon', encode='pert_iname')
    TrainTotal, ValTotal = utils.train_val_split(df, 0.8)
    if GROUP_WELLS:
        gTDF = TrainTotal.groupby('Metadata_labels')
        TrainLoaders.append(DataloaderTrainV6(TrainTotal, nr_cells=initial_cells, nr_sets=nr_sets, groupDF=gTDF))
        gVDF = ValTotal.groupby('Metadata_labels')
        ValLoaders.append(DataloaderTrainV6(ValTotal, nr_cells=initial_cells, nr_sets=nr_sets, groupDF=gVDF, compensator=72))
    else:
        TrainLoaders.append(DataloaderTrainV6(TrainTotal, nr_cells=initial_cells, nr_sets=nr_sets))
        ValLoaders.append(DataloaderTrainV6(ValTotal, nr_cells=initial_cells, nr_sets=nr_sets))


train_sets = torch.utils.data.ConcatDataset(TrainLoaders)
trainloader = data.DataLoader(train_sets, batch_size=BS, shuffle=True, collate_fn=utils.my_collate,
                           drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)
val_sets = torch.utils.data.ConcatDataset(ValLoaders)
valloader = data.DataLoader(val_sets, batch_size=BS, shuffle=True, collate_fn=utils.my_collate,
                         drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)

print(f'Using {len(trainloader.dataset.cumulative_sizes)} plates.')
if input('Continue? [y/n]') == 'y':
    pass
else:
    run.finish()
    raise Warning('Stopping run')

# %% Setup models
model = MLPsum(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim, k=kFilters)
print(model)
print([p.numel() for p in model.parameters() if p.requires_grad])
total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total number of parameters:', total_parameters)
if torch.cuda.is_available():
    model.cuda()
# %% Setup optimizer
optimizer = optim.AdamW(model.parameters(),
                        lr=lr)
loss_func = losses.SupConLoss(distance=distances.CosineSimilarity())


# Configure WandB parameters, so that they are saved with each training
config = wandb.config
config.kFilters = kFilters
config.nr_cells = nr_cells
config.latent_dim = latent_dim
config.output_dim = output_dim
config.nr_sets = nr_sets
config.batch_size = BS
config.trueBS = true_BS
config.epochs = epochs
config.learning_rate = lr
config.optimizer = optimizer
config.architecture = model

wandb.watch(model, loss_func, log='all', log_freq=5, log_graph=True) # log_freq: log every x batches

# %% Start training
print(utils.now() + "Start training")
best_val = np.inf

for e in range(epochs):
    time.sleep(0.5)
    model.train()
    tr_loss = 0.0

    print("Training epoch")
    for idx, (points, labels) in enumerate(tqdm(trainloader)):
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

        # RESET NR_CELLS
        if isinstance(nr_cells, tuple):
            CELLS = int(np.random.normal(nr_cells[0], nr_cells[1], 1))
            for z in range(len(TrainLoaders)):
                trainloader.dataset.datasets[z].nr_cells = CELLS


    tr_loss /= (idx+1)
    wandb.log({"Train Loss": tr_loss}, step=e)  # send data to wandb.ai

    #print(utils.now() + f"Epoch {e}. Training loss: {tr_loss}.")
    #continue

    # Validation
    model.eval()
    val_loss = 0.0
    time.sleep(0.5)
    print('Validation epoch')
    time.sleep(0.5)
    with torch.no_grad():
        for i, (points, labels) in enumerate(tqdm(valloader)):
            points, labels = points.to(device), labels.to(device)

            # Retrieve feature embeddings
            feats, _ = model(points)
            # Calculate loss
            val_loss_tmp = loss_func(feats, labels)
            # add the loss to running variable
            val_loss += val_loss_tmp.item()

            # RESET NR_CELLS
            if isinstance(nr_cells, tuple):
                CELLS = int(np.random.normal(nr_cells[0], nr_cells[1], 1))
                for z in range(len(ValLoaders)):
                    valloader.dataset.datasets[z].nr_cells = CELLS

    val_loss /= (i+1)
    wandb.log({"Val loss": val_loss}, step=e)  # send data to wandb.ai

    print(utils.now() + f"Epoch {e}. Training loss: {tr_loss}. Validation loss: {val_loss}.")

    if val_loss < best_val:
        best_val = val_loss
        print('Writing best val model checkpoint')
        print('best val loss:{}'.format(best_val))

        torch.save(model.state_dict(), os.path.join(run.dir, f'model_bestval_{save_name_extension}'))

    print('Creating model checkpoint...')
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tr_loss': tr_loss,
        'val_loss': val_loss,
    }, os.path.join(run.dir, f'general_ckpt_{save_name_extension}'))

print(utils.now() + 'Finished training')
run.finish()


