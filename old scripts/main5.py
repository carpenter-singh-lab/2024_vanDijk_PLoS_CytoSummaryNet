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
BS = 80  # batch size as passed to the model (nr of wells that you sample per batch)
true_BS = int(nr_sets*BS) # true number of sets that the model will see per batch
print(f'True batch size is {true_BS}')
# nr of cells sampled from each well (no more than 1200 found in compound plates)
# May be good to look at the average number of cells per well for each dataset or tune as hyperparameter
nr_cells = 500
input_dim = 1324
kFilters = 4  # times DIVISION of filters in model
latent_dim = 1028  # test with 128
output_dim = 512

# TODO UNDER DEVELOPMENT
PR_epoch = False
group_by_feature = 'Metadata_labels' # FOR PR CALCULATION

weight_decay = 0


#%% Load all Stain2 data
rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/Stain2'
metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv', index_col=False)
plateDirs = [x[0] for x in os.walk(rootDir)][1:]

# TODO Currently you can only use 3 plates for training
plates = ['BR00112197standard_FS', 'BR00112199_FS', 'BR00112197repeat_FS']

plateDirs = [x for x in plateDirs if any(substr in x for substr in plates)]

TrainLoaders = []
ValLoaders = []
for i, pDir in enumerate(plateDirs):
    C_metadata = utils.addDataPathsToMetadata(rootDir, metadata, pDir)
    df = utils.filterData(C_metadata, 'negcon', encode='pert_iname')
    TrainTotal, ValTotal = utils.train_val_split(df, 0.8)
    trainset = DataloaderTrainV5(TrainTotal, nr_cells=nr_cells, nr_sets=nr_sets)
    valset = DataloaderTrainV5(ValTotal, nr_cells=nr_cells, nr_sets=nr_sets)
    TrainLoaders.append(data.DataLoader(trainset, batch_size=BS, shuffle=True, collate_fn=utils.my_collate,
                               drop_last=False, pin_memory=False, num_workers=NUM_WORKERS))
    ValLoaders.append(data.DataLoader(valset, batch_size=BS, shuffle=True, collate_fn=utils.my_collate,
                             drop_last=False, pin_memory=False, num_workers=NUM_WORKERS))


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
config.architecture = save_name_extension
config.optimizer = optimizer

wandb.watch(model, loss_func, log='all', log_freq=10)

# %% Start training
print(utils.now() + "Start training")
best_val = np.inf


for e in range(epochs):
    model.train()
    tr_loss = 0.0
    dataloader_iterator2 = iter(TrainLoaders[1])
    dataloader_iterator3 = iter(TrainLoaders[2])
    print("Training epoch")
    for idx, (points1, labels1) in enumerate(tqdm(TrainLoaders[0])):
        points1, labels1 = points1.to(device), labels1.to(device)
        try:
            points2, labels2 = next(dataloader_iterator2)
        except:
            dataloader_iterator2 = iter(TrainLoaders[1])
            points2, labels2 = next(dataloader_iterator2)
        try:
            points3, labels3 = next(dataloader_iterator3)
        except:
            dataloader_iterator3 = iter(TrainLoaders[2])
            points3, labels3 = next(dataloader_iterator3)

        # Retrieve feature embeddings
        feats1, _ = model(points1)
        feats2, _ = model(points2)
        feats3, _ = model(points3)
        # Calculate loss
        tr_loss_tmp1 = loss_func(feats1, labels1)
        tr_loss_tmp2 = loss_func(feats2, labels2)
        tr_loss_tmp3 = loss_func(feats3, labels3)

        # add the loss to running variable
        tr_loss_tmp = (tr_loss_tmp1 + tr_loss_tmp2 + tr_loss_tmp3)/3
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
    dataloader_iterator2 = iter(ValLoaders[1])
    dataloader_iterator3 = iter(ValLoaders[2])
    with torch.no_grad():
        for i, (points1, labels1) in enumerate(tqdm(ValLoaders[0])):
            points1, labels1 = points1.to(device), labels1.to(device)

            try:
                points2, labels2 = next(dataloader_iterator2)
            except:
                dataloader_iterator2 = iter(ValLoaders[1])
                points2, labels2 = next(dataloader_iterator2)
            try:
                points3, labels3 = next(dataloader_iterator3)
            except:
                dataloader_iterator3 = iter(ValLoaders[2])
                points3, labels3 = next(dataloader_iterator3)

            # Retrieve feature embeddings
            feats1, _ = model(points1)
            feats2, _ = model(points2)
            feats3, _ = model(points3)
            # Calculate loss
            val_loss_tmp1 = loss_func(feats1, labels1)
            val_loss_tmp2 = loss_func(feats2, labels2)
            val_loss_tmp3 = loss_func(feats3, labels3)
            # add the loss to running variable
            val_loss_tmp = (val_loss_tmp1 + val_loss_tmp2 + val_loss_tmp3)/3
            val_loss += val_loss_tmp.item()

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


