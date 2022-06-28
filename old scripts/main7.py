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
from pytorch_metric_learning import regularizers, losses, distances

# Custom libraries
import wandb
from networks.SimpleMLPs import MLP, MLPmean, MLPsum
from dataloader_pickles import DataloaderTrainV4, DataloaderTrainV5, DataloaderEvalV5
import utils

from utils import CalculatePercentReplicating
import utils_benchmark

#%%
run = wandb.init(project="FeatureAggregation", mode='online', tags=['Generalized Models'])  # 'dryrun'

# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
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


lr = 1e-3  # learning rate
epochs = 50  # maximum number of epochs
nr_sets = 3 # number of sets you sample per well
BS = 72  # batch size as passed to the model (nr of wells that you sample per batch)
true_BS = int(nr_sets*BS) # true number of sets that the model will see per batch
print(f'True batch size is {true_BS}')
# nr of cells sampled from each well (no more than 1200 found in compound plates)
# May be good to look at the average number of cells per well for each dataset or tune as hyperparameter
initial_cells = 800
nr_cells = (800, 200)  # INTEGER or TUPLE (median, std) for gaussian // (1100, 300)
input_dim = 1324
kFilters = 1  # times DIVISION of filters in model
latent_dim = 1028  # test with 128
output_dim = 512

weight_decay = 0

#%% Load all Stain2 data
rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/Stain2'
metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv', index_col=False)
plateDirs = [x[0] for x in os.walk(rootDir)][1:]

plates = ['BR00112197binned_FS', 'BR00112199_FS', 'BR00112203_FS', 'BR00113818_FS', 'BR00113820_FS']

plateDirs = [x for x in plateDirs if any(substr in x for substr in plates)]

TrainLoaders = []
ValLoaders = []
for i, pDir in enumerate(plateDirs):
    C_metadata = utils.addDataPathsToMetadata(rootDir, metadata, pDir)
    df = utils.filterData(C_metadata, 'negcon', encode='pert_iname')
    TrainTotal, ValTotal = utils.train_val_split(df, 0.8)
    trainset = DataloaderTrainV5(TrainTotal, nr_cells=initial_cells, nr_sets=nr_sets)
    valset = DataloaderTrainV5(ValTotal, nr_cells=initial_cells, nr_sets=nr_sets)
    TrainLoaders.append(data.DataLoader(trainset, batch_size=BS, shuffle=True, collate_fn=utils.my_collate,
                                        drop_last=False, pin_memory=False, num_workers=NUM_WORKERS))
    ValLoaders.append(data.DataLoader(valset, batch_size=BS, shuffle=True, collate_fn=utils.my_collate,
                                      drop_last=False, pin_memory=False, num_workers=NUM_WORKERS))


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
                        lr=lr,
                        weight_decay=weight_decay)
R = regularizers.SparseCentersRegularizer(num_classes=len(TrainTotal['Metadata_labels'].unique()), centers_per_class=nr_sets)
loss_func = losses.SupConLoss(distance=distances.CosineSimilarity(), embedding_regularizer=R)


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

wandb.watch(model, loss_func, log='all', log_freq=len(TrainLoaders[0]), log_graph=True) # log_freq: log every x batches

# %% Start training
print(utils.now() + "Start training")
best_val = np.inf

for e in range(epochs):
    time.sleep(0.5)
    model.train()
    tr_loss = 0.0

    print("Training epoch")
    for idx, data in enumerate(tqdm(zip(TrainLoaders[0], TrainLoaders[1], TrainLoaders[2], TrainLoaders[3], TrainLoaders[4]), total=len(TrainLoaders[0]))):
        points, labels = [d[0] for d in data], [d[1] for d in data]
        points, labels = [x.to(device) for x in points], [y.to(device) for y in labels]

        temp_losses = []
        for P, L in zip(points, labels):
            # Retrieve feature embeddings
            feats, _ = model(P)
            # Calculate loss
            temp_losses.append(loss_func(feats, L))

        tr_loss_tmp = sum(temp_losses)/len(TrainLoaders)
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
                TrainLoaders[z].dataset.nr_cells = CELLS

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
        for i, data in enumerate(tqdm(zip(ValLoaders[0], ValLoaders[1], ValLoaders[2], ValLoaders[3], ValLoaders[4]), total=len(ValLoaders[0]))):
            points, labels = [d[0] for d in data], [d[1] for d in data]
            points, labels = [x.to(device) for x in points], [y.to(device) for y in labels]

            temp_losses = []
            for P, L in zip(points, labels):
                # Retrieve feature embeddings
                feats, _ = model(P)
                # Calculate loss
                temp_losses.append(loss_func(feats, L))

            val_loss_tmp = sum(temp_losses)/len(ValLoaders)
            # add the loss to running variable
            val_loss += val_loss_tmp.item()

            # RESET NR_CELLS
            if isinstance(nr_cells, tuple):
                CELLS = int(np.random.normal(nr_cells[0], nr_cells[1], 1))
                for z in range(len(TrainLoaders)):
                    ValLoaders[z].dataset.nr_cells = CELLS

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


