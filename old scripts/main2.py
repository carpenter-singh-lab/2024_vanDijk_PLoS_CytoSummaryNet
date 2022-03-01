## Standard libraries
import os
import random
import numpy as np
import time

## tqdm for loading bars
from tqdm import tqdm

## PyTorch
import torch
import torch.utils.data as data
import torch.optim as optim

# Losses
from pytorch_metric_learning import miners, losses

# Custom libraries
import wandb
from networks.SimpleMLPs import MLP
from dataloader_pickles import DataloaderTrain, DataloaderTrainV2
from utils import info_nce_loss, now


run = wandb.init(project="FeatureAggregation", tags=['OnlyControls', 'WellPosition'], mode='online')  # 'dryrun'

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
save_name_extension = 'simpleMLP_OnlyControls'  # extension of the saved model, this string will be added to the script name

model_name = 'model_' + save_name_extension
print(model_name)
lr = 0.0001  # learning rate
epochs = 50  # maximum number of epochs
BS = 32  # batch size
temperature = 0.07  # TODO - check around what value this should
latent_dim = 1938
weight_decay = 1e-4

# %% Load all data
rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/OnlyControls_noORF_noOWells'

plateDirTrain1 = 'DataLoader_Plate00117010_OnlyControls'
tdir1 = os.path.join(rootDir, plateDirTrain1)
plateDirTrain2 = 'DataLoader_Plate00117011_OnlyControls'
tdir2 = os.path.join(rootDir, plateDirTrain2)
plateDirVal1 = 'DataLoader_Plate00117012_OnlyControls'
vdir1 = os.path.join(rootDir, plateDirVal1)
plateDirVal2 = 'DataLoader_Plate00117013_OnlyControls'
vdir2 = os.path.join(rootDir, plateDirVal2)

filenamesTrain1 = [os.path.join(tdir1, file) for file in os.listdir(tdir1)]
filenamesTrain2 = [os.path.join(tdir2, file) for file in os.listdir(tdir2)]
filenamesTrain1.sort()
filenamesTrain2.sort()
TrainTotal = filenamesTrain1 + filenamesTrain2

filenamesVal1 = [os.path.join(vdir1, file) for file in os.listdir(vdir1)]
filenamesVal1.sort()
filenamesVal2 = [os.path.join(vdir2, file) for file in os.listdir(vdir2)]
filenamesVal2.sort()
ValTotal = filenamesVal1 + filenamesVal2

trainset = DataloaderTrainV2(TrainTotal, label_type='well_name')
valset = DataloaderTrainV2(ValTotal, label_type='well_name')

train_loader = data.DataLoader(trainset, batch_size=BS, shuffle=True,
                               drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
val_loader = data.DataLoader(valset, batch_size=BS, shuffle=True,
                             drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

# %% Setup models
model = MLP(nr_channels=1938, latent_dim=latent_dim)
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
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=epochs,
                                                    eta_min=lr / 50)
loss_func = losses.NPairsLoss()
miner = miners.MultiSimilarityMiner(epsilon=0.1)

# Configure WandB parameters, so that they are saved with each training
config = wandb.config
config.batch_size = BS
config.epochs = epochs
config.learning_rate = lr
config.architecture = 'simpleMLP'
config.optimizer = optimizer

wandb.watch(model)
# %% Start training
print(now() + "Start training")
best_val = np.inf

for e in range(epochs):
    model.train()
    tr_loss = 0.0
    print("Training epoch")
    for idx, (points, labels) in enumerate(tqdm(train_loader)):
        #points = torch.cat([x for x in points])  # this puts plate 1 in the first BS entries [:BS, :] and plate 2 in the second BS entries [BS:, :]
        points = points.float()  # necessary for model and loss computations
        points, labels = points.to(device), labels.to(device)

        # Retrieve feature embeddings
        feats, _ = model(points)

        # Calculate loss
        hard_pairs = miner(feats, labels)
        tr_loss_tmp = loss_func(feats, labels, hard_pairs)  # embeddings, labels, indices_tuple, ref_emb, ref_labels):

        #tr_loss_tmp, _ = info_nce_loss(points, model, temperature)

        # add the loss to running variable
        tr_loss += tr_loss_tmp.item()

        # Adam
        optimizer.zero_grad()
        tr_loss_tmp.backward()
        optimizer.step()

    tr_loss /= (idx+1)
    wandb.log({"Train Loss": tr_loss}, step=e)  # send data to wandb.ai

    # Validation
    model.eval()
    val_loss = 0.0
    time.sleep(0.5)
    print('Validation epoch')
    time.sleep(0.5)
    with torch.no_grad():
        for i, points in enumerate(tqdm(val_loader)):
            points = torch.cat([x for x in points])
            points = points.float()  # necessary for model and loss computations
            points = points.to(device)
            # Calculate InfoNCE loss
            val_loss_tmp, _ = info_nce_loss(points, model, temperature)

            # add the loss to running variable
            val_loss += val_loss_tmp.item()

    val_loss /= (i+1)
    wandb.log({"Val loss": val_loss}, step=e)  # send data to wandb.ai
    print(now() + f"Epoch {e}. Training loss: {tr_loss}. Validation loss: {val_loss}.")

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

print(now() + 'Finished training')
run.finish()


