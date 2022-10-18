## Standard libraries
import os
import glob
import pandas as pd

## Seeds
import random
import numpy as np

## PyTorch
import torch

# Custom libraries
from utils import CalculatePercentReplicating
from pycytominer.operations.transform import RobustMAD
from pycytominer import feature_select
import utils_benchmark
import utils

NUM_WORKERS = 0
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# Set random seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

#%% Load data
train_val_split = 1.0
dataset_name = 'LINCS'
percent_matching = True

rootdir = f'/Users/rdijk/Documents/ProjectFA/Phase2/Data/profiles'
plateDirs = glob.glob(os.path.join(rootdir, '*.csv'))
platenames = [x.split('/')[-1][:-4] for x in plateDirs]

metadata_dir = '/Users/rdijk/Documents/ProjectFA/Phase2/Data/metadata'
barcode_platemap = pd.read_csv(os.path.join(metadata_dir, 'barcode_platemap.csv'), index_col=False)
barcode_platemap = barcode_platemap[barcode_platemap['Assay_Plate_Barcode'].isin(platenames)]

platemaps = barcode_platemap['Plate_Map_Name'].tolist()
platenames = barcode_platemap['Assay_Plate_Barcode'].tolist()

plateDirs = ['/Users/rdijk/Documents/ProjectFA/Phase2/Data/profiles/'+x+'.csv' for x in platenames]

I = platemaps.index('C-7161-01-LM6-013')
plateDirs.pop(I)
platemaps.pop(I)
platenames.pop(I)

repurposing_info = pd.read_csv('/Users/rdijk/Documents/ProjectFA/Phase2/Data/metadata/repurposing_info_long.tsv',
                               sep='\t', header=0, low_memory=False)
repurposing_info = repurposing_info.rename(columns={'broad_id': 'broad_sample'})

# %% Calculate benchmark on training set used
print('Calculating Benchmark')
group_by_feature = 'Metadata_labels'

corr_replicating_df = pd.DataFrame()
AllResultsDF = pd.DataFrame()

full_df = []
for i, filepath in enumerate(plateDirs):
    platestring = filepath.split('/')[-1].split('_')[-1][:-4]
    metadata = pd.read_csv(os.path.join(metadata_dir, 'platemap', platemaps[i]+'.txt'), sep='\t')
    metadata['moa'] = list(metadata.broad_sample.map(dict(zip(repurposing_info.broad_sample, repurposing_info.moa))))
    df = pd.read_csv(filepath, index_col=False, low_memory=False)
    if percent_matching:
        df['Metadata_labels'] = metadata['moa']
    else:
        df['Metadata_labels'] = metadata['broad_sample']

    # Select features
    features = df.iloc[:, 2:-1]

    scaler = RobustMAD(epsilon=0)
    fitted_scaler = scaler.fit(features)
    features = fitted_scaler.transform(features)

    features['Metadata_labels'] = df['Metadata_labels']
    features['Metadata_pert_iname'] = metadata['broad_sample']
    features['Metadata_moa'] = metadata['moa']
    features['Metadata_mmoles_per_liter'] = metadata['mmoles_per_liter']
    features = features[np.logical_and(features['Metadata_mmoles_per_liter']>9, features['Metadata_mmoles_per_liter']<11)]
    features = features.dropna(subset=['Metadata_labels'])  # remove potentially unannotated broad_samples or MoAs
    print(features.shape)
    full_df.append(features)

full_df = pd.concat(full_df)
full_df = full_df.dropna(axis=1, how='any')
all_features = feature_select(full_df.iloc[:, :-4], operation=["variance_threshold", "correlation_threshold",
                                               "drop_na_columns", "blocklist"])
full_df = pd.concat([all_features, full_df.iloc[:, -4:]], axis=1)

### MAP analysis
split = int(len(full_df.index) * train_val_split)

AP = utils.CalculateMAP(full_df, 'cosine_similarity', 'Metadata_labels', percent_matching)
print('Total mean:', AP.AP.mean())
# COMMENT OUT FOR MOA PREDICTION
# print(f'Training samples || mean:{AP.iloc[:split, :].mean()}\n',
#       AP.iloc[:split, :].groupby('compound').mean().sort_values(by='AP', ascending=False).to_markdown())
# print(f'Validation samples || mean:{AP.iloc[split:, :].mean()}\n',
#       AP.iloc[split:, :].groupby('compound').mean().sort_values(by='AP', ascending=False).to_markdown())

f = open(f'outputs/MAPs/{dataset_name}_mAP_BM.txt', 'a')
f.write('\n')
f.write('\n')
f.write(f'Plate: {platestring}')
f.write('\n')
f.write('Total mean:' + str(AP.AP.mean()))
f.write('\n')
f.write(f'Training samples || mean:{AP.iloc[:split, :].mean()}\n' + AP.iloc[:split, :].groupby(
    'compound').mean().sort_values(by='AP', ascending=False).to_markdown())
f.write('\n')
f.write(f'Validation samples || mean:{AP.iloc[split:, :].mean()}\n' + AP.iloc[split:, :].groupby(
    'compound').mean().sort_values(by='AP', ascending=False).to_markdown())
f.close()


if percent_matching:
    AllResultsDF = pd.concat([AllResultsDF, pd.DataFrame({'mAP BM': [AP.AP.mean()] })])
else:
    AllResultsDF = pd.concat([AllResultsDF, pd.DataFrame({'Training mAP BM': [AP.iloc[:split, 1].mean()],
                                                          'Validation mAP BM': [AP.iloc[split:, 1].mean()] })])

#%%
PLATES = [x.split('/')[-1].split('_')[-1][:-4] for x in plateDirs]
PLATES.sort()
AllResultsDF['plate'] = '_'.join(PLATES)

AllResultsDF = AllResultsDF.set_index('plate')

print(AllResultsDF.round(2).to_markdown())
if not percent_matching:
    AllResultsDF.to_csv(f'/Users/rdijk/Documents/ProjectFA/Phase2/Results/Benchmarks/{dataset_name}_BM.csv')
else:
    AllResultsDF.to_csv(f'/Users/rdijk/Documents/ProjectFA/Phase2/Results/Benchmarks/{dataset_name}_BM_MOA.csv')
