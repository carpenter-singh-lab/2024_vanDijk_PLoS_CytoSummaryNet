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
import sys
from sklearn.decomposition import PCA
from pycytominer.operations.transform import RobustMAD

NUM_WORKERS = 0
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

# Set random seed for reproducibility
manualSeed = 512
# manualSeed = random.randint(1,10000) # use if you want new results
print("Random Seed:", manualSeed)
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

#%%
dataset_name = 'Stain2'
MAPfilename = f'MAP_dummy_cells_{dataset_name}'

path = r'../wandb/run-20220317_174244-31lhybhb/files'  # wellAugment_noCP

dummy_cell_frac = None
dummy_feature_frac = 0.1

#%%
models = os.listdir(path)
fullpath = os.path.join(path, model_name)
if 'ckpt' in model_name:
    model.load_state_dict(torch.load(fullpath)['model_state_dict'])
else:
    model.load_state_dict(torch.load(fullpath))
model.eval()
#%% Load all data
rootDir = fr'/Users/rdijk/PycharmProjects/featureAggregation/datasets/{dataset_name}'
PLATES = [x[0] for x in os.walk(rootDir)][1:]

AllResultsDF = pd.DataFrame()
for plate in PLATES:
    metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv', index_col=False)
    platestring = plate.split('_')[-2]
    print('Calculating results for: ' + platestring)
    metadata = utils.addDataPathsToMetadata(rootDir, metadata, plate)
    df_prep = utils.filterData(metadata, 'negcon', encode='pert_iname')
    Total, _ = utils.train_val_split(df_prep, 1.0, sort=False)
    valset = DataloaderEvalV5(Total)
    loader = data.DataLoader(valset, batch_size=1, shuffle=False,
                                   drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)

    #%% Create feature dataframes
    MLP_profiles = pd.DataFrame()

    print('Calculating Features')
    with torch.no_grad():
        for idx, (points, labels) in enumerate(tqdm(loader)):

            if dummy_cell_frac:
                dummy_cells = torch.rand((points.shape[0], int(points.shape[1]*dummy_cell_frac), points.shape[2]))
                points = torch.cat([points, dummy_cells], dim=1)

            if dummy_feature_frac:
                dummy_features = np.random.choice(points.shape[2], int(dummy_feature_frac*points.shape[2]),
                                                  replace=False)
                points[:, :, dummy_features] = 0

            feats, _ = model(points)
            c1 = pd.concat([pd.DataFrame(feats), pd.Series(labels)], axis=1)
            MLP_profiles = pd.concat([MLP_profiles, c1])




    #%% Rename columns
    MLP_profiles.columns = [f"f{x}" for x in range(MLP_profiles.shape[1] - 1)] + ['Metadata_labels']
    print('MLP_profiles shape: ', MLP_profiles.shape)

    if mAPcalc:
        split = int(MLP_profiles.shape[0] * 0.8)
        MLP_profiles['Metadata_pert_iname'] = list(metadata[metadata['control_type']!='negcon']['pert_iname'])
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

    save_name = f"{dataset_name}_{platestring}"  # "TVsplit_allWells_gene_nR3"  ||  "TVsplit_OnlyControls_well_nR3"
    group_by_feature = 'Metadata_labels'

    n_replicates = [int(round(MLP_profiles['Metadata_labels'].value_counts().mean()))]
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

    AllResultsDF = pd.concat([AllResultsDF, pd.DataFrame({'Training mAP model': AP.iloc[:split, 1].mean(),
                                             'Validation mAP model': AP.iloc[split:, 1].mean(),
                                             'PR model': corr_replicating_df['Percent_Replicating'].astype(float)})])

PLATES = [x.split('_')[-2] for x in PLATES]
AllResultsDF['plate'] = PLATES

# Load BM results
BMdf = pd.read_csv(f'/Users/rdijk/Documents/ProjectFA/Benchmarks/{dataset_name}_BM.csv')
AllResultsDF = AllResultsDF.merge(BMdf, on='plate')

AllResultsDF = AllResultsDF.set_index('plate')
cols = AllResultsDF.columns.tolist()
cols = [cols[0], cols[3], cols[1], cols[4], cols[2], cols[5]]
AllResultsDF = AllResultsDF[cols]
print(AllResultsDF.round(2).to_markdown())
