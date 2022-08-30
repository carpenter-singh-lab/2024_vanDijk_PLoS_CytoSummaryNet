## Standard libraries
import os

import sklearn.mixture
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

## Seeds
import random
import numpy as np

## PyTorch
import torch
import torch.utils.data as data

# Custom libraries
from networks.SimpleMLPs import MLPsumV2
from dataloader_pickles import DataloaderEvalV5, DataloaderTrainV6
import utils
from utils import CalculatePercentReplicating
import utils_benchmark
import sys

##
NUM_WORKERS = 0
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

# Set random seed for reproducibility
manualSeed = 42
# manualSeed = random.randint(1,10000) # use if you want new results
print("Random Seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

## %% Load model
save_name_extension = 'model_bestval_simpleMLP_V1'  # extension of the saved model
model_name = save_name_extension
print('Loading:', model_name)

input_dim = 1783  # 1324
kFilters = 1/2
latent_dim = 2048
output_dim = 2048
model = MLPsumV2(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim,
                 k=kFilters, dropout=0, cell_layers=1,
                 proj_layers=2, reduction='sum')
if torch.cuda.is_available():
    model.cuda()

save_features_to_csv = True
mAPcalc = True
train_val_split = 1.0

percent_matching = False  # If false calculate percent replicating

if percent_matching:
    encoding_label = 'moa'
    mAP_label = 'Metadata_moa'
    bestMOAs = pd.DataFrame()
else:
    encoding_label = 'pert_iname'
    mAP_label = 'Metadata_pert_iname'

dataset_name = 'LINCS'
MAPfilename = f'mAP_{dataset_name}_test_1'

path = r'wandb/latest-run/files'

#path = r'wandb/run-20220517_164505-2q0md5h8/files'  # Stain234 15 plates
#path = r'wandb/run-20220503_161251-29xy65t4/files' # Stain234 12 plates
#path = r'wandb/run-20220505_221947-1m1zas58/files' # Stain234 12 plates outliers

models = os.listdir(path)
fullpath = os.path.join(path, model_name)

if 'ckpt' in model_name:
    model.load_state_dict(torch.load(fullpath)['model_state_dict'])
else:
    model.load_state_dict(torch.load(fullpath))
model.eval()
# %% Load all data
rootDir = fr'/Users/rdijk/PycharmProjects/featureAggregation/datasets/{dataset_name}'
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

repurposing_info = pd.read_csv('/Users/rdijk/Documents/ProjectFA/Phase2/Data/metadata/repurposing_info_long.tsv',
                               sep='\t', header=0, low_memory=False)
repurposing_info = repurposing_info.rename(columns={'broad_id': 'broad_sample'})

# Initialize variables
AllResultsDF = pd.DataFrame()
ALLCELLS = []
average_perturbation_map = {}
plate_loadings = pd.DataFrame()

Loaders = []
metadata = []
for i, plate in enumerate(plateDirs):
    platestring = plate.split('_')[-1]
    print('Getting data from: ' + platestring)

    C_plate_map = pd.read_csv(os.path.join(metadata_dir, 'platemap', platemaps[i]+'.txt'), sep='\t')
    C_metadata = utils.addDataPathsToMetadata(rootDir, C_plate_map, plate)
    # Filter the data and create numerical labels
    df = utils.filterData(C_metadata, 'negcon', encode='broad_sample', mode='LINCS')
    df = df[np.logical_and(df['mmoles_per_liter'] > 9, df['mmoles_per_liter'] < 11)]

    Total, _ = utils.train_val_split(df, 1.0, sort=True)

    metadata.append(df)
    Loaders.append(DataloaderEvalV5(Total, preprocess=None))  # TODO WATCH OUT FOR THIS "preprocess" PARAMETER

metadata = pd.concat(metadata).reset_index(drop=True)

sets = torch.utils.data.ConcatDataset(Loaders)
loader = data.DataLoader(sets, batch_size=1, shuffle=False,
                                drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)


# %% Create feature dataframes
MLP_profiles = pd.DataFrame()

NRCELLS = []
print('Calculating Features')

with torch.no_grad():
    for idx, (points, labels) in enumerate(tqdm(loader)):
        feats, _ = model(points)
        # Append everything to dataframes
        c1 = pd.concat([pd.DataFrame(feats), pd.Series(labels)], axis=1)
        MLP_profiles = pd.concat([MLP_profiles, c1])

## % Rename columns and normalize features
# Rename feature columns
MLP_profiles.columns = [f"f{x}" for x in range(MLP_profiles.shape[1] - 1)] + ['Metadata_labels']
print('MLP_profiles shape: ', MLP_profiles.shape)
MLP_profiles.reset_index(drop=True, inplace=True)
# %% Save all the dataframes to .csv files!
if save_features_to_csv:
    MLP_profiles.to_csv(f'outputs/{dataset_name}/MLP_profiles_{platestring}.csv', index=False)

if mAPcalc:
    split = int(MLP_profiles.shape[0] * train_val_split)
    MLP_profiles['Metadata_pert_iname'] = list(metadata.dropna(subset=['broad_sample'])['broad_sample'])
    MLP_profiles['Metadata_moa'] = list(metadata.dropna(subset=['broad_sample']).broad_sample.map(dict(zip(repurposing_info.broad_sample, repurposing_info.moa))))
    print('Dropping ', MLP_profiles.shape[0] - MLP_profiles.dropna().reset_index(drop=True).shape[0], 'rows due to NaNs')
    MLP_profiles = MLP_profiles.dropna().reset_index(drop=True)
    print('New size:', MLP_profiles.shape)

    AP = utils.CalculateMAP(MLP_profiles, 'cosine_similarity',
                            groupby=mAP_label, percent_matching=percent_matching)

    print('Total mean mAP:', AP.AP.mean(), '\nTotal mean precision at R:', AP['precision at R'].mean())
    print(f'Training samples || mean:{AP.iloc[:split,1].mean()}\n', AP.iloc[:split,:].groupby('compound').mean().sort_values(by='AP',ascending=False).to_markdown())
    print(f'Validation samples || mean:{AP.iloc[split:,1].mean()}\n', AP.iloc[split:,:].groupby('compound').mean().sort_values(by='AP',ascending=False).to_markdown())

    # WRITE TO FILE
    f = open(f'outputs/MAPs/{MAPfilename}.txt', 'a')
    f.write('\n')
    f.write('\n')
    f.write(f'Plate: {platestring}')
    f.write('\n')
    f.write('Total mean:' + str(AP.AP.mean()))
    f.write('\n')
    f.write(f'Training samples || mean:{AP.iloc[:split,1].mean()}\n' + AP.iloc[:split,:].groupby('compound').mean().sort_values(by='AP',ascending=False).to_markdown())
    f.write('\n')
    f.write(f'Validation samples || mean:{AP.iloc[split:,1].mean()}\n' + AP.iloc[split:,:].groupby('compound').mean().sort_values(by='AP',ascending=False).to_markdown())
    f.close()

    for z in range(len(AP)):
        try:
            average_perturbation_map[AP.loc[z, 'compound']] += AP.loc[z, 'AP']
        except:
            average_perturbation_map[AP.loc[z, 'compound']] = AP.loc[z, 'AP']
##%
if percent_matching:
    AllResultsDF = pd.concat([AllResultsDF, pd.DataFrame({'mAP model': [AP.iloc[:, 1].mean()]
                                                          })])
    sorted_dictionary = {k: [v] for k, v in sorted(average_perturbation_map.items(), key=lambda item: item[1], reverse=True)}
    bestMOAs = pd.concat([bestMOAs, pd.DataFrame(sorted_dictionary)])
else:
    AllResultsDF = pd.concat([AllResultsDF, pd.DataFrame({'Training mAP model': [AP.iloc[:split, 1].mean()],
                                                          'Validation mAP model': [AP.iloc[split:, 1].mean()]
                                                          })])


PLATES = [x.split('_')[-1] for x in plateDirs]
PLATES.sort()
AllResultsDF['plate'] = '_'.join(PLATES)

# Load BM results
if percent_matching:
    BMdf = pd.read_csv(f'/Users/rdijk/Documents/ProjectFA/Phase2/Results/Benchmarks/{dataset_name}_BM_MOA.csv')
else:
    BMdf = pd.read_csv(f'/Users/rdijk/Documents/ProjectFA/Phase2/Results/Benchmarks/{dataset_name}_BM.csv')
AllResultsDF = AllResultsDF.merge(BMdf, on='plate')

AllResultsDF = AllResultsDF.set_index('plate')
cols = AllResultsDF.columns.tolist()
if percent_matching:
    cols = [cols[0], cols[1]]
else:
    cols = [cols[0], cols[2], cols[1], cols[3]]

AllResultsDF = AllResultsDF[cols]
print(AllResultsDF.round(2).to_markdown())

if percent_matching:
    AllResultsDF.to_csv(f'/Users/rdijk/Documents/ProjectFA/Phase2/Results/Tests/TestResults_{dataset_name}.csv')
else:
    AllResultsDF.to_csv(f'/Users/rdijk/Documents/ProjectFA/Phase2/Results/Tests/TestResults_replicating_{dataset_name}.csv')

if percent_matching:
    bestMOAs.to_csv(f'/Users/rdijk/Documents/ProjectFA/Phase2/Results/Tests/bestMOAs_{dataset_name}.csv')
