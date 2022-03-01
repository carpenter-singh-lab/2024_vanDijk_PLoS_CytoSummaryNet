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
from networks.SimpleMLPs import MLP, MLPadapt
from dataloader_pickles import DataloaderTrainV4, DataloaderTrainV5, DataloaderEvalV5
import utils

from utils import CalculatePercentReplicating
import utils_benchmark

#%%
run = wandb.init(project="FeatureAggregation", mode='online', tags=['Generalized Models', 'FS'])  # 'dryrun'

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
save_name_extension = 'simpleMLP_V1'  # extension of the saved model, this string will be added to the script name

model_name = 'model_' + save_name_extension
print(model_name)


lr = 1e-3  # learning rate
epochs = 50  # maximum number of epochs
nr_sets = 2 # number of sets you sample per well
BS = 64  # batch size as passed to the model (nr of wells that you sample per batch)
true_BS = int(nr_sets*BS) # true number of sets that the model will see per batch
print(f'True batch size is {true_BS}')
# nr of cells sampled from each well (no more than 1200 found in compound plates)
# May be good to look at the average number of cells per well for each dataset or tune as hyperparameter
nr_cells = 600
input_dim = 838
kFilters = 4  # times DIVISION of filters in model
latent_dim = 1028  # test with 128
output_dim = 512

PR_epoch = False # TODO UNDER DEVELOPMENT
group_by_feature = 'Metadata_labels' # FOR PR CALCULATION

weight_decay = 0

# %% Load all data
rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/CPJUMP1'
DATATYPE = 'FS'
# Set paths for all training/validation files
plateDirTrain1 = f'DataLoader_BR00117010_{DATATYPE}'
tdir1 = os.path.join(rootDir, plateDirTrain1)
plateDirTrain2 = f'DataLoader_BR00117011_{DATATYPE}'
tdir2 = os.path.join(rootDir, plateDirTrain2)
plateDirVal1 = f'DataLoader_BR00117012_{DATATYPE}'
vdir1 = os.path.join(rootDir, plateDirVal1)
plateDirVal2 = f'DataLoader_BR00117013_{DATATYPE}'
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
AllData, _ = utils.train_val_split(df, 1)

trainset = DataloaderTrainV5(TrainTotal, nr_cells=nr_cells, nr_sets=nr_sets)
valset = DataloaderTrainV5(ValTotal, nr_cells=nr_cells, nr_sets=nr_sets)
PRset = DataloaderEvalV5(AllData)

train_loader = data.DataLoader(trainset, batch_size=BS, shuffle=True, collate_fn=utils.my_collate,
                               drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)
val_loader = data.DataLoader(valset, batch_size=BS, shuffle=True, collate_fn=utils.my_collate,
                             drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)
PR_loader = data.DataLoader(PRset, batch_size=1)

# %% Setup models
model = MLP(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim, k=kFilters)
#model = MLPadapt(pool_size=800, latent_dim=latent_dim, output_dim=output_dim, k=kFilters)
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
loss_func = losses.SupConLoss(distance=distances.SNRDistance())

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
config.architecture = 'simpleMLP'
config.optimizer = optimizer

wandb.watch(model, loss_func, log='all', log_freq=10)

# %% Start training
print(utils.now() + "Start training")
best_val = np.inf


for e in range(epochs):
    model.train()
    tr_loss = 0.0
    print("Training epoch")
    for idx, (points, labels) in enumerate(tqdm(train_loader)):

        # points, labels = torch.cat([x for x in points]), torch.cat([x for x in labels])  # this puts plate 1 in the first BS entries [:BS, :] and plate 2 in the second BS entries [BS:, :]
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
        for i, (points, labels) in enumerate(tqdm(val_loader)):
            points, labels = points.to(device), labels.to(device)
            # Retrieve feature embeddings
            feats, _ = model(points)
            # Calculate loss
            val_loss_tmp = loss_func(feats, labels)
            # add the loss to running variable
            val_loss += val_loss_tmp.item()

    val_loss /= (i+1)
    wandb.log({"Val loss": val_loss}, step=e)  # send data to wandb.ai

    #%% TODO UNDER DEVELOPMENT:
    if PR_epoch:
        time.sleep(0.5)
        print('Percent Replicating epoch')
        time.sleep(0.5)
        MLP_profiles = pd.DataFrame()
        with torch.no_grad():
            for i, (points, labels) in enumerate(tqdm(PR_loader)):
                points, labels = points.to(device), labels.to(device)
                # Retrieve feature embeddings
                feats, _ = model(points)
                # Append everything to dataframes
                c1 = pd.concat([pd.DataFrame(feats), pd.Series(labels)], axis=1)
                MLP_profiles = pd.concat([MLP_profiles, c1])

        # TODO find a metric (perhaps PR) that describes generalization better, possibilities include:
        # TODO 1. PR over an entire dataset that has never been seen by the model before

        n_replicatesT = int(round(MLP_profiles['Metadata_labels'].value_counts().mean()))
        n_samples = 10000
        dataframes = [MLP_profiles]
        nReplicates = [6]
        corr_replicating_df = pd.DataFrame()
        for plates, nR in zip(dataframes, nReplicates):
            temp_df = CalculatePercentReplicating(plates, group_by_feature, nR, n_samples)
            corr_replicating_df = pd.concat([corr_replicating_df, temp_df], ignore_index=True)
        PR = corr_replicating_df['Percent Replicating'].astype(float)
        wandb.log({"PR": PR}, step=e)  # send data to wandb.ai


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


