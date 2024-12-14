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
src.utils import CalculatePercentReplicating
from pycytominer.operations.transform import RobustMAD
from pycytominer import feature_select
import src.utils_benchmark as utils_benchmark
import utils

NUM_WORKERS = 0
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# Set random seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

#%% Load data
dataset_name = 'Stain2'
percent_matching = False

rootdir = f'/Users/rdijk/Documents/Data/ProcessedData/{dataset_name}/profiles'
plateDirs = glob.glob(os.path.join(rootdir, '*.csv'))

metadata = pd.read_csv('/inputs/cpg0001_metadata/JUMP-MOA_compound_platemap_with_metadata.csv',
                       index_col=False)
commonFeatureNames = pd.read_csv('/Users/rdijk/Documents/Data/RawData/CommonFeatureNames.csv', index_col=False)
commonFeatureNames = [x.split('.')[1] for x in commonFeatureNames.iloc[:, 0]]

# %% Calculate Percent Replicating on training set used
print('Calculating Percent Replicating')
group_by_feature = 'Metadata_labels'
n_samples = 10000
n_replicates = 4

corr_replicating_df = pd.DataFrame()
AllResultsDF = pd.DataFrame()

for filepath in plateDirs:
    platestring = filepath.split('/')[-1].split('_')[-1][:-4]

    if filepath.endswith('BR00112200.csv') or filepath.endswith('BR00115130.csv'):
        continue
    df = pd.read_csv(filepath, index_col=False, low_memory=False)
    if percent_matching:
        df['Metadata_labels'] = metadata['moa']
    else:
        df['Metadata_labels'] = metadata['pert_iname']
    df['Metadata_pert_iname'] = metadata['pert_iname']
    df['control_type'] = metadata['control_type']

    # Select 1324 features
    features = df[commonFeatureNames]

    scaler = RobustMAD(epsilon=0)
    fitted_scaler = scaler.fit(features)
    features = fitted_scaler.transform(features)
    features = features.dropna(axis=1, how='any')
    features = feature_select(features, operation=["variance_threshold", "correlation_threshold",
                                                   "drop_na_columns", "blocklist"])

    features['Metadata_labels'] = df['Metadata_labels']
    features['Metadata_pert_iname'] = df['Metadata_pert_iname']
    features = features[metadata.control_type != 'negcon']

    ### MAP analysis
    split = int(len(features.index) * 0.8)
    assert split == 288
    AP = utils.CalculateMAP(features, 'cosine_similarity', 'Metadata_labels', percent_matching)
    # print('Total mean:', AP.AP.mean())
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

    # PERCENT REPLICATING
    temp_df = CalculatePercentReplicating(features, group_by_feature, n_replicates, n_samples, platestring, percent_matching)
    corr_replicating_df = pd.concat([corr_replicating_df, temp_df], ignore_index=True)

    if percent_matching:
        AllResultsDF = pd.concat([AllResultsDF, pd.DataFrame({'mAP BM': AP.iloc[:, 1].mean(),
                                                              'PR BM': temp_df['Percent_Replicating'].astype(float)})])
    else:
        AllResultsDF = pd.concat([AllResultsDF, pd.DataFrame({'Training mAP BM': AP.iloc[:split, 1].mean(),
                                                              'Validation mAP BM': AP.iloc[split:, 1].mean(),
                                                              'PR BM': temp_df['Percent_Replicating'].astype(float)})])

utils_benchmark.distribution_plot(df=corr_replicating_df, output_file=f'Benchmark_{dataset_name}', metric="Percent Replicating")


#%%
PLATES = [x.split('/')[-1].split('_')[-1][:-4] for x in plateDirs]
if dataset_name == 'Stain3':
    PLATES.remove('BR00115130')  # Remove outlier plate
if dataset_name == 'Stain2':
    PLATES.remove('BR00112200')  # Remove outlier plate
AllResultsDF['plate'] = PLATES

AllResultsDF = AllResultsDF.set_index('plate')

print(AllResultsDF.round(2).to_markdown())
if not percent_matching:
    AllResultsDF.to_csv(f'/Users/rdijk/Documents/ProjectFA/Benchmarks/{dataset_name}_BM.csv')
else:
    AllResultsDF.to_csv(f'/Users/rdijk/Documents/ProjectFA/Benchmarks/{dataset_name}_BM_MOA.csv')
