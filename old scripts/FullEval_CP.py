## Standard libraries
import os
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
from networks.SimpleMLPs import MLP, MLPsum, MLPmean
from dataloader_pickles import DataloaderEvalV5, DataloaderTrainV6
import utils
from utils import CalculatePercentReplicating
import utils_benchmark
import glob
import sys
from sklearn.decomposition import PCA
from pycytominer.operations.transform import RobustMAD
from pycytominer import feature_select


NUM_WORKERS = 0
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Set random seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# %% Load model
save_name_extension = 'general_ckpt_simpleMLP_V1'  # extension of the saved model // model_bestval_
model_name = save_name_extension
print('Loading:', model_name)

input_dim = 1324 # 1938 // 838 // 800
kFilters = 1  # times DIVISION of filters in model
latent_dim = 1024
output_dim = 512
model = MLPsum(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim, k=kFilters)

if torch.cuda.is_available():
    model.cuda()

save_features_to_csv = False
evaluate_point_distributions = False
mAPcalc = True
MAPfilename = 'MAP_wellAugment_outlierPlates_CP'

path = r'../wandb/run-20220321_104829-3qidy2i9/files'  # 3 plates
#path = r'wandb/run-20220318_173720-1zk0l561/files' # 2 plates
#path = r'wandb/run-20220307_214134-1qowwhuq/files' # Stain2exp3V2 max

models = os.listdir(path)
fullpath = os.path.join(path, model_name)
if 'ckpt' in model_name:
    model.load_state_dict(torch.load(fullpath)['model_state_dict'])
else:
    model.load_state_dict(torch.load(fullpath))
model.eval()
# %% Load all data
USE_MODEL = False

#plates_selection = ['BR00112198', 'BR00112204', 'BR00112201']
#plates_selection = ['BR00112202', 'BR00112197standard', 'BR00113818', 'BR00113819', 'BR00112197repeat', 'BR00112197binned']
plates_selection = ['BR00112203', 'BR00112199', 'BR00113820', 'BR00113821']
platestring = 'BM_outlier_plates_CP'

metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv',
                       index_col=False)

AllResultsDF = pd.DataFrame()

if USE_MODEL:
    print('Using model')
    rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/Stain2'
    PLATES = [x[0] for x in os.walk(rootDir)][1:]
    plateDirs = [x for x in PLATES if any(substr in x for substr in plates_selection)]

    TrainLoaders = []
    for i, pDir in enumerate(plateDirs):
        C_metadata = utils.addDataPathsToMetadata(rootDir, metadata, pDir)
        df_prep = utils.filterData(metadata, 'negcon', encode='pert_iname')
        Total, _ = utils.train_val_split(df_prep, 1.0, sort=False)

        TrainLoaders.append(DataloaderEvalV5(Total, norm=False))

    train_sets = torch.utils.data.ConcatDataset(TrainLoaders)
    loader = data.DataLoader(train_sets, batch_size=1, shuffle=False,
                                       drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)

    # %% Create feature dataframes
    MLP_profiles = pd.DataFrame()

    print('Calculating Features')
    with torch.no_grad():
        for idx, (points, labels) in enumerate(tqdm(loader)):
            # labels = labels.to(device)[0, ...]
            # points = points.to(device)[0, ...]
            feats, _ = model(points)
            # Append everything to dataframes
            c1 = pd.concat([pd.DataFrame(feats), pd.Series(labels)], axis=1)
            MLP_profiles = pd.concat([MLP_profiles, c1])

    # %% Rename columns and normalize features
    # Rename feature columns
    MLP_profiles.columns = [f"f{x}" for x in range(MLP_profiles.shape[1] - 1)] + ['Metadata_labels']
    print('MLP_profiles shape: ', MLP_profiles.shape)

else:
    print('Calculating benchmark')
    rootdir = '/Users/rdijk/Documents/Data/ProcessedData/Stain2/profiles'
    PLATES = glob.glob(os.path.join(rootdir, '*.csv'))
    plateDirs = [x for x in PLATES if any(substr in x for substr in plates_selection)]  # validation
    commonFeatureNames = pd.read_csv('/Users/rdijk/Documents/Data/RawData/CommonFeatureNames.csv', index_col=False)
    commonFeatureNames = [x.split('.')[1] for x in commonFeatureNames.iloc[:, 0]]
    negconsStain = ["A11", "B18", "D17", "D19", "E07", "E08", "F07", "F24", "G20", "G23", "H10", "I12", "I13", "J01",
                    "J21", "K24", "M05", "M06", "N06", "N22", "O14", "O19", "P02", "P11"]
    MLP_profiles = pd.DataFrame()
    for i, pDir in enumerate(plateDirs):
        df = pd.read_csv(pDir, index_col=False, low_memory=False)
        # Select 1324 features
        features = df[commonFeatureNames]

        scaler = RobustMAD(epsilon=0)
        fitted_scaler = scaler.fit(features)
        features = fitted_scaler.transform(features)
        features = features[~df.Metadata_Well.isin(negconsStain)].reset_index(drop=True)  # remove negcons
        MLP_profiles = pd.concat([MLP_profiles, features])
    MLP_profiles = MLP_profiles.dropna(axis=1, how='any')
    MLP_profiles = feature_select(MLP_profiles, operation=["variance_threshold", "correlation_threshold",
                                                           "drop_na_columns", "blocklist"])

NPLATES = len(plateDirs)

# %% Analyze feature distributions
# for df in [plate1df, plate2df, plate3df, plate4df]:
if evaluate_point_distributions:
    nrRows = 16
    df_MLP = MLP_profiles.iloc[:, :-1]  # Only pass the features

    utils.featureCorrelation(df_MLP, nrRows)
    utils.compoundCorrelation(df_MLP, nrRows)
    utils.createUmap(MLP_profiles, 30)  # need the labels for Umap

if mAPcalc:
    split = int(MLP_profiles.shape[0]*0.8)
    MLP_profiles['Metadata_pert_iname'] = NPLATES * list(metadata[metadata['control_type']!='negcon']['pert_iname'])
    AP = utils.CalculateMAP(MLP_profiles, 'cosine_similarity', groupby='Metadata_pert_iname')
    print('Total mean:', AP.AP.mean())
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
# %% Calculate Percent Replicating on training set used
print('Calculating Percent Replicating')

save_name = f"Stain2_{platestring}"  # "TVsplit_allWells_gene_nR3"  ||  "TVsplit_OnlyControls_well_nR3"
group_by_feature = 'Metadata_pert_iname'

n_replicates = [int(round(MLP_profiles['Metadata_pert_iname'].value_counts().mean()))]
n_samples = 10000

dataframes = [MLP_profiles]

descriptions = [platestring]
print('nReplicates used: ', n_replicates)

corr_replicating_df = pd.DataFrame()
for plates, nR, desc in zip(dataframes, n_replicates, descriptions):
    temp_df = CalculatePercentReplicating(plates, group_by_feature, nR, n_samples, desc)
    corr_replicating_df = pd.concat([corr_replicating_df, temp_df], ignore_index=True)

print(corr_replicating_df[['Description', 'Percent_Replicating']].to_markdown(index=False))

utils_benchmark.distribution_plot(df=corr_replicating_df, output_file=f"{save_name}_PR.png",
                                  metric="Percent Replicating")

corr_replicating_df['Percent_Replicating'] = corr_replicating_df['Percent_Replicating'].astype(float)

plot_corr_replicating_df = (
    corr_replicating_df.rename(columns={'Modality': 'Perturbation'})
        .drop(columns=['Null_Replicating', 'Value_95', 'Replicating'])
)

AllResultsDF = pd.concat([AllResultsDF, pd.DataFrame({'Training mAP model': AP.iloc[:288,1].mean(),
                                         'Validation mAP model': AP.iloc[288:,1].mean(),
                                         'PR model': corr_replicating_df['Percent_Replicating'].astype(float)})])

