# Standard libraries
import os
import random
import numpy as np
import time
import pandas as pd

# tqdm for loading bars
from tqdm import tqdm

# PyTorch
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

##% Set hyperparameters
lr = 1e-6  # learning rate
epochs = 50  # maximum number of epochs
nr_sets = 4  # number of sets you sample per well
BS = 18  # 54 for grouped // batch size as passed to the model (nr of wells that you sample per batch)
true_BS = int(nr_sets * BS)  # true number of sets that the model will see per batch
print(f'True batch size is {true_BS}')
initial_cells = 2000
nr_cells = (initial_cells, 900)  # INTEGER or TUPLE (median, std) for gaussian // (1100, 300)
input_dim = 1324
kFilters = 1 / 2  # times DIVISION of filters in model
latent_dim = 2048
output_dim = 2048
dropout = 0.0

#tr_fraction_list = [0.1, 0.2, 0.4, 0.8]
tr_fraction_list = [0.8]

for tr_fraction in tr_fraction_list:
    save_name_extension = 'finetuned'  # extension of the saved model, specify architecture used
    model_name = f'model_{save_name_extension}_{tr_fraction}'
    print(model_name)

    ##% Load all data
    rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/Stain5'
    metadata = pd.read_csv('/inputs/cpg0001_metadata/JUMP-MOA_compound_platemap_with_metadata.csv',
                           index_col=False)
    plateDirs = [x[0] for x in os.walk(rootDir)][1:]

    plates = ['BR00120536_FS', 'BR00120270_FS', 'BR00120532_FS', 'BR00120526', 'BR00120526confocal']
    plateDirs = [x for x in plateDirs if any(substr in x for substr in plates)]

    TrainLoaders = []
    ValLoaders = []
    for i, pDir in enumerate(plateDirs):
        C_metadata = utils.addDataPathsToMetadata(rootDir, metadata, pDir)
        df = utils.filterData(C_metadata, 'negcon', encode='pert_iname')
        TrainTotal, _ = utils.train_val_split(df, tr_fraction)
        ValTotal, _ = utils.train_val_split(df, 1.0)

        gTDF = TrainTotal.groupby('Metadata_labels')
        trainset = DataloaderTrainV7(TrainTotal, nr_cells=initial_cells, nr_sets=nr_sets, groupDF=gTDF)
        TrainLoaders.append(data.DataLoader(trainset, batch_size=BS, shuffle=True, collate_fn=utils.my_collate,
                                            drop_last=False, pin_memory=False, num_workers=NUM_WORKERS))

        valset = DataloaderEvalV5(ValTotal)
        ValLoaders.append(data.DataLoader(valset, batch_size=1, shuffle=False,
                                       drop_last=False, pin_memory=False, num_workers=NUM_WORKERS))

    print(f'\nLoading {len(TrainLoaders)} plates. Did you check the training loop?')

    ##% Setup models
    model = MLPsumV2(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim,
                     k=kFilters, dropout=0, cell_layers=1,
                     proj_layers=2, reduction='sum')

    if torch.cuda.is_available():
        model.cuda()
    # %% Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_func = losses.SupConLoss(distance=distances.CosineSimilarity(), temperature=0.1)

    ##% Load model
    # path = r'run-20220517_164505-2q0md5h8' # Stain234 15 plates
    # path = r'run-20220503_161251-29xy65t4' # Stain234 12 plates
    # path = r'run-20220505_221947-1m1zas58'  # Stain234 12 plates outliers

    run_name = r'run-20220517_164505-2q0md5h8'
    dir_name = f'wandb/{run_name}/files'
    print(f'Resuming training with {run_name}...')
    model.load_state_dict(torch.load(f'{dir_name}/model_bestval_simpleMLP_V1'))
    optimizer.load_state_dict(torch.load(f'{dir_name}/general_ckpt_simpleMLP_V1')['optimizer_state_dict'])

    ##% Start training
    print(utils.now() + "Start training")
    best_val = 0

    for e in range(epochs):
        time.sleep(0.5)
        model.train()
        tr_loss = 0.0

        print("Training epoch")
        for idx, platepoints in enumerate(tqdm(zip(TrainLoaders[0], TrainLoaders[1], TrainLoaders[2], TrainLoaders[3], TrainLoaders[4]), total=len(TrainLoaders[0]))):
            points, labels = [d[0] for d in platepoints], [d[1] for d in platepoints]
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
                while CELLS < 100 or CELLS % 2 != 0:
                    CELLS = int(np.random.normal(nr_cells[0], nr_cells[1], 1))
                for z in range(len(TrainLoaders)):
                    TrainLoaders[z].dataset.nr_cells = CELLS


        tr_loss /= (idx+1)

        # Validation
        model.eval()

        time.sleep(0.5)
        print('Validation epoch')
        time.sleep(0.5)

        temp_losses = []
        temp_mAPs = []
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

        print(utils.now() + f"Epoch {e}. Train loss: {tr_loss}. Val loss: {val_loss}. Val mAP: {val_mAP}")

        if val_mAP > best_val:
            best_val = val_mAP
            print('Writing best val model checkpoint')
            print('best val mAP:{}'.format(best_val))

            torch.save(model.state_dict(), os.path.join(dir_name,  model_name))

    print(utils.now() + f'Finished training with tr_fraction {tr_fraction}')
