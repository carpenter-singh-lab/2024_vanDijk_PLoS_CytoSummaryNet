##% Standard libraries
import os
import random
import numpy as np
import time
import pandas as pd
pd.set_option('display.max_columns', 10)

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

# argument parser
import argparse

##%%
def train_model_LINCS(args):
    run = wandb.init(project="FeatureAggregation", mode=args.wandb_mode, tags=['LINCS'])  # 'dryrun'
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

    lr = args.lr  # learning rate
    epochs = args.epochs  # maximum number of epochs
    nr_sets = args.nr_sets  # number of sets you sample per well
    BS = args.bs  # 54 for grouped // batch size as passed to the model (nr of wells that you sample per batch)
    true_BS = int(nr_sets*BS) # true number of sets that the model will see per batch
    print(f'True batch size is {true_BS}')
    initial_cells = args.initial_cells
    nr_cells = (initial_cells, 800)  # INTEGER or TUPLE (median, std) for gaussian // (1100, 300)
    input_dim = args.model_input_size  # 1324
    kFilters = args.kfilters  # times DIVISION of filters in model
    latent_dim = 2048
    output_dim = 2048
    dropout = 0.0

    load_model = False
    weight_decay = 'AdamW default'

    #%% Load all data
    rootDir = r'datasets/LINCS'  # path to datasets
    plateDirs = [x[0] for x in os.walk(rootDir)][1:]
    platenames = [x.split('_')[-1] for x in plateDirs]

    metadata_dir = args.metadata_path  # path to metadata
    barcode_platemap = pd.read_csv(os.path.join(metadata_dir, 'barcode_platemap.csv'), index_col=False)
    barcode_platemap = barcode_platemap[barcode_platemap['Assay_Plate_Barcode'].isin(platenames)]

    repurposing_info = pd.read_csv(os.path.join(metadata_dir, 'repurposing_info_long.tsv'), index_col=False,
                                   low_memory=False, sep='\t', usecols=["broad_id", "pert_iname", "moa"])
    repurposing_info = repurposing_info.rename(columns={"broad_id": "broad_sample"})
    repurposing_info = repurposing_info.drop_duplicates()

    platemaps = barcode_platemap['Plate_Map_Name'].tolist()
    platenames = barcode_platemap['Assay_Plate_Barcode'].tolist()

    plateDirs = ['DataLoader_'+x for x in platenames]

    I = platemaps.index('C-7161-01-LM6-013')
    plateDirs.pop(I)
    platemaps.pop(I)
    platenames.pop(I)

    bigdf = []
    for i, pDir in enumerate(plateDirs):
        C_plate_map = pd.read_csv(os.path.join(metadata_dir, 'platemap', platemaps[i]+'.txt'), sep='\t')
        C_metadata = utils.addDataPathsToMetadata(rootDir, C_plate_map, pDir)
        df = C_metadata[np.logical_and(C_metadata['mmoles_per_liter'] > 9, C_metadata['mmoles_per_liter'] < 11)]
        bigdf.append(df)
    bigdf = pd.merge(pd.concat(bigdf), repurposing_info, on='broad_sample', how='left')
    bigdf = utils.filterData(bigdf, 'negcon', encode='pert_iname', mode='LINCS')
    shape1 = bigdf.shape[0]
    bigdf.dropna(inplace=True)  # drop all compounds without annotations for pert_iname (and moa)
    shape2 = bigdf.shape[0]
    print("Removed", shape1-shape2, "wells due to missing annotation of pert_iname and moa.")
    bigdf = bigdf[bigdf.Metadata_labels.duplicated(keep=False)]
    shape3 = bigdf.shape[0]
    print("Removed", shape2-shape3, "unique compound wells.")
    print('Using', shape3, "wells")
    Total, _ = utils.train_val_split(bigdf, 1.0, sort=True)
    gTDF = Total.groupby('Metadata_labels')
    TrainDataset = DataloaderTrainV7(Total, nr_cells=initial_cells, nr_sets=nr_sets, groupDF=gTDF)
    ValDataset = DataloaderEvalV5(Total)

    trainloader = data.DataLoader(TrainDataset, batch_size=BS, shuffle=True, collate_fn=utils.my_collate,
                                        drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)
    valloader = data.DataLoader(ValDataset, batch_size=1, shuffle=False,
                                   drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)

    print(f'\nLoading {len(plateDirs)} plates with {len(TrainDataset)} unique compounds. Do you want to proceed?')
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
                trainloader.dataset.nr_cells = CELLS

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
                if points.shape[1] == 1:
                    continue
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

if __name__=='__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Train a simple MLP to aggregate and cluster replicate compound profiles '
                                                 'near each other and non-replicate compound profiles far away.',
                                     fromfile_prefix_chars='@')

    # Optional positional argument
    parser.add_argument('metadata_path', nargs='?', const='aws_scripts/metadata/platemaps/2016_04_01_a549_48hr_batch1',
                        type=str,
                        help='Specify the path where the barcode_platemap and platemaps directory are located.')
    # Optional positional argument
    parser.add_argument('wandb_mode', nargs='?', const='online', type=str,
                        help='Sync the data with the wandb server with "online" or run offline with "dryrun".')
    # Optional positional argument
    parser.add_argument('model_input_size', nargs='?', const=1781, type=int,
                        help='Number of single cell features.')
    # Optional positional argument
    parser.add_argument('lr', nargs='?', const=5e-4, type=float,
                        help='AdamW learning rate')
    # Optional positional argument
    parser.add_argument('epochs', nargs='?', const=100, type=int,
                        help='Number of epochs to train the model')
    # Optional positional argument
    parser.add_argument('nr_sets', nargs='?', const=4, type=int,
                        help='Number of data augmented single cell feature sets to create from each unique compound')
    # Optional positional argument
    parser.add_argument('bs', nargs='?', const=18, type=int,
                        help='batch size')
    # Optional positional argument
    parser.add_argument('initial_cells', nargs='?', const=1500, type=int,
                        help='Initial mean of the gaussian used to sample the number of cells that are sampled for all'
                             ' sets of each unique compound')
    # Optional positional argument
    parser.add_argument('kfilters', nargs='?', const=1/2, type=float,
                        help='Times division of the number of filters in the hidden model layers')

    # Parse arguments
    args = parser.parse_args()

    print("Argument values:")
    print('metadata path:', args.metadata_path)
    print('wandb mode:', args.wandb_mode)
    print('model input size:', args.model_input_size)
    print('lr:', args.lr)
    print('epochs:', args.epochs)
    print('nr_sets:', args.nr_sets)
    print('bs:', args.bs)
    print('initial_cells:', args.initial_cells)
    print('kfilters:', args.kfilters)

    train_model_LINCS(args)
