##% Standard libraries
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
from networks.SimpleMLPs import MLPsumV2
from dataloader_pickles import DataloaderTrainV7, DataloaderEvalV5
import utils

##%%
run = wandb.init(project="FeatureAggregation", mode='online', tags=['LINCS'])  # 'dryrun'
wandb.define_metric("Val loss", summary="min")
wandb.define_metric("Val mAP", summary="max")
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

##%% Set hyperparameters
save_name_extension = 'simpleMLP_V1'  # extension of the saved model, specify architecture used

model_name = 'model_' + save_name_extension
print(model_name)

lr = 5e-4  # learning rate
epochs = 100  # maximum number of epochs
nr_sets = 4  # number of sets you sample per well
BS = 18  # 54 for grouped // batch size as passed to the model (nr of wells that you sample per batch)
true_BS = int(nr_sets*BS) # true number of sets that the model will see per batch
print(f'True batch size is {true_BS}')
initial_cells = 1500
nr_cells = (initial_cells, 800)  # INTEGER or TUPLE (median, std) for gaussian // (1100, 300)
input_dim = 1783  # 1324
kFilters = 1/2  # times DIVISION of filters in model
latent_dim = 2048
output_dim = 2048
dropout = 0.0

load_model = False
weight_decay = 'AdamW default'

#%% Load all data
rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/LINCS'
plateDirs = [x[0] for x in os.walk(rootDir)][1:]
platenames = [x.split('_')[-1] for x in plateDirs]

metadata_dir = '/Users/rdijk/Documents/ProjectFA/Phase2/Data/metadata'
barcode_platemap = pd.read_csv(os.path.join(metadata_dir, 'barcode_platemap.csv'), index_col=False)
barcode_platemap = barcode_platemap[barcode_platemap['Assay_Plate_Barcode'].isin(platenames)]

platemaps = barcode_platemap['Plate_Map_Name'].tolist()
platenames = barcode_platemap['Assay_Plate_Barcode'].tolist()

plateDirs = ['/Users/rdijk/PycharmProjects/featureAggregation/datasets/LINCS/DataLoader_'+x for x in platenames]

I = platemaps.index('C-7161-01-LM6-013')
plateDirs.pop(I)
platemaps.pop(I)
platenames.pop(I)

TrainLoaders = []
ValLoaders = []
for i, pDir in enumerate(plateDirs):
    C_plate_map = pd.read_csv(os.path.join(metadata_dir, 'platemap', platemaps[i]+'.txt'), sep='\t')
    C_metadata = utils.addDataPathsToMetadata(rootDir, C_plate_map, pDir)
    df = utils.filterData(C_metadata, 'negcon', encode='broad_sample', mode='LINCS')
    df = df[np.logical_and(df['mmoles_per_liter'] > 9, df['mmoles_per_liter'] < 11)]
    Total, _ = utils.train_val_split(df, 1.0, sort=True)
    gTDF = Total.groupby('Metadata_labels')
    TrainLoaders.append(DataloaderTrainV7(Total, nr_cells=initial_cells, nr_sets=nr_sets, groupDF=gTDF))
    ValLoaders.append(DataloaderEvalV5(Total))

train_sets = torch.utils.data.ConcatDataset(TrainLoaders)
trainloader = data.DataLoader(train_sets, batch_size=BS, shuffle=True, collate_fn=utils.my_collate,
                                    drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)
val_sets = torch.utils.data.ConcatDataset(ValLoaders)
valloader = data.DataLoader(val_sets, batch_size=1, shuffle=False,
                               drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)

print(f'\nLoading {len(TrainLoaders)} plates. Did you check the training loop?')
if input('Continue? [y/n]') == 'y':
    pass
else:
    run.finish()
    raise Warning('Stopping run')

# %% Setup models
model = MLPsumV2(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim,
                 k=kFilters, dropout=0, cell_layers=1,
                 proj_layers=2, reduction='sum')

print(model)
print([p.numel() for p in model.parameters() if p.requires_grad])
total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total number of parameters:', total_parameters)
if torch.cuda.is_available():
    model.cuda()
# %% Setup optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr)
loss_func = losses.SupConLoss(distance=distances.CosineSimilarity(), temperature=0.1) #[0.01 - 0.2]
#loss_func = losses.FastAPLoss(num_bins=10) # bins = [5, 10, 20, 40, 80]
#%% Load model
if load_model:
    run_name = r'run-20220516_182720-26261k0a'
    print(f'Resuming training with {run_name}...')
    checkpoint = torch.load(f'wandb/{run_name}/files/general_ckpt_simpleMLP_V1')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    E = checkpoint['epoch']
else:
    E = 0

#%% Configure WandB parameters, so that they are saved with each training
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
config.dropout = dropout
config.architecture = model
config.load_model = load_model

# TrainLoaders = utils.filter_noisy_data(plateDirs, rootDir, model, config)
wandb.watch(model, loss_func, log='all', log_freq=10)  # log_freq: log every x batches

# %% Start training
print(utils.now() + "Start training")
best_val = 0

for e in range(E, E+epochs):
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
            while CELLS < 100 or CELLS % 2 != 0:
                CELLS = int(np.random.normal(nr_cells[0], nr_cells[1], 1))
            for z in range(len(trainloader.dataset.datasets)):
                trainloader.dataset.datasets[z].nr_cells = CELLS

    tr_loss /= (idx+1)
    wandb.log({"Train Loss": tr_loss}, step=e)  # send data to wandb.ai

    # Validation
    model.eval()

    time.sleep(0.5)
    print('Validation epoch')
    time.sleep(0.5)

    with torch.no_grad():
        MLP_profiles = torch.tensor([], dtype=torch.float32)
        MLP_labels = torch.tensor([], dtype=torch.int16)
        for (points, labels) in tqdm(valloader):
            feats, _ = model(points)
            # Append everything to dataframes
            MLP_profiles = torch.cat([MLP_profiles, feats])
            MLP_labels = torch.cat([MLP_labels, labels])

        # Calculate loss
        val_loss_tmp = loss_func(MLP_profiles, MLP_labels)
        # assign the loss (no running variable because this is calculated over the entire dataset)
        val_loss = val_loss_tmp.item()

        # Calculate mAP
        MLP_profiles = pd.concat(
            [pd.DataFrame(MLP_profiles.detach().numpy()), pd.Series(MLP_labels.detach().numpy())], axis=1)
        MLP_profiles.columns = [f"f{x}" for x in range(MLP_profiles.shape[1] - 1)] + ['Metadata_label']
        AP = utils.CalculateMAP(MLP_profiles, 'cosine_similarity',
                                groupby='Metadata_label', percent_matching=False)
        # assign the mAP (no running variable because this is calculated over the entire dataset)
        val_mAP = AP.AP.mean()

    print(utils.now() + f"Epoch {e}. Train loss: {tr_loss}. Val loss: {val_loss}. Val mAP: {val_mAP}")

    if val_mAP > best_val:
        best_val = val_mAP
        print('Writing best val model checkpoint')
        print('best val mAP:{}'.format(best_val))

        torch.save(model.state_dict(), os.path.join(run.dir, f'model_bestval_{save_name_extension}'))

    wandb.log({"Val loss": val_loss, "Val mAP": val_mAP, "best_val_mAP": best_val}, step=e)  # send data to wandb.ai

    print('Creating model checkpoint...')
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tr_loss': tr_loss,
        'val_loss': val_loss,
        'val_mAP': val_mAP,
    }, os.path.join(run.dir, f'general_ckpt_{save_name_extension}'))

print(utils.now() + 'Finished training')
run.finish()

