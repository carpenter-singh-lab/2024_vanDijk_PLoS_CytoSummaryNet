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

<<<<<<< HEAD
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.spatial import distance
from scipy.cluster import hierarchy

# %% Load model
#save_name_extension = 'general_ckpt_simpleMLP_V1'  # extension of the saved model
save_name_extension = 'model_bestval_simpleMLP_V1'  # extension of the saved model
#save_name_extension = 'model_finetuned_0.8'
model_name = save_name_extension
print('Loading:', model_name)

input_dim = 1324  # 1938 // 838 // 800
kFilters = 1/2  # times DIVISION of filters in model
=======
## %% Load model
save_name_extension = 'model_bestval_simpleMLP_V1'  # extension of the saved model
model_name = save_name_extension
print('Loading:', model_name)

input_dim = 1324  # 1324
kFilters = 1/2
>>>>>>> fac3077 (Updated scripts for preprocessing, training, and evaluating of LINCS data.)
latent_dim = 2048
output_dim = 2048
model = MLPsumV2(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim,
                 k=kFilters, dropout=0, cell_layers=1,
                 proj_layers=2, reduction='sum')
if torch.cuda.is_available():
    model.cuda()

save_features_to_csv = True
evaluate_point_distributions = False
mAPcalc = True
PRcalc = False

percent_matching = False  # If false calculate percent replicating

if percent_matching:
    encoding_label = 'moa'
    mAP_label = 'Metadata_moa'
    bestMOAs = pd.DataFrame()
else:
    encoding_label = 'pert_iname'
    mAP_label = 'Metadata_pert_iname'

dataset_name = 'Stain4'
<<<<<<< HEAD
MAPfilename = f'MAP_Stain234_15_FINAL_{dataset_name}'

path = r'wandb/latest-run/files'
=======
MAPfilename = f'mAP_{dataset_name}_test_1'

path = r'wandb/latest-run/files'

>>>>>>> fac3077 (Updated scripts for preprocessing, training, and evaluating of LINCS data.)
#path = r'wandb/run-20220517_164505-2q0md5h8/files'  # Stain234 15 plates
#path = r'wandb/run-20220503_161251-29xy65t4/files' # Stain234 12 plates
#path = r'wandb/run-20220505_221947-1m1zas58/files' # Stain234 12 plates outliers

<<<<<<< HEAD
#path = r'wandb/run-20220427_191823-och8e6jv/files'  # Stain34 6 plates


models = os.listdir(path)
fullpath = os.path.join(path, model_name)

#%% TODO TODO  remove this stuff
# fullpath = os.path.join('/Users/rdijk/Documents/ProjectFA/InterpretabilityAnalysis/LearningCovarianceMatrices/models',
#                      'general_ckpt_simpleMLP_V1_sphered')
#%% TODO TODO remove this stuff on top
=======
models = os.listdir(path)
fullpath = os.path.join(path, model_name)

>>>>>>> fac3077 (Updated scripts for preprocessing, training, and evaluating of LINCS data.)
if 'ckpt' in model_name:
    model.load_state_dict(torch.load(fullpath)['model_state_dict'])
else:
    model.load_state_dict(torch.load(fullpath))
model.eval()
# %% Load all data
rootDir = fr'/Users/rdijk/PycharmProjects/featureAggregation/datasets/{dataset_name}'
PLATES = [x[0] for x in os.walk(rootDir)][1:]

AllResultsDF = pd.DataFrame()
ALLCELLS = []


<<<<<<< HEAD

=======
>>>>>>> fac3077 (Updated scripts for preprocessing, training, and evaluating of LINCS data.)
average_perturbation_map = {}
plate_loadings = pd.DataFrame()
for plate in PLATES:
    metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv', index_col=False)

    platestring = plate.split('_')[-2]
    print('Calculating results for: ' + platestring)
    metadata = utils.addDataPathsToMetadata(rootDir, metadata, plate)

    # Filter the data and create numerical labels
    df_prep = utils.filterData(metadata, 'negcon', encode=encoding_label)
    # Add all data to one DF
    Total, _ = utils.train_val_split(df_prep, 1.0, sort=False)

    valset = DataloaderEvalV5(Total, preprocess=None)  # TODO WATCH OUT FOR THIS "preprocess" PARAMETER
    loader = data.DataLoader(valset, batch_size=1, shuffle=False,
                                   drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)

    # %% Create feature dataframes
    MLP_profiles = pd.DataFrame()

    NRCELLS = []
    print('Calculating Features')

    with torch.no_grad():
        for idx, (points, labels) in enumerate(tqdm(loader)):
            # NRCELLS.append(points.shape[1])
            # continue
            feats, _ = model(points)  # TODO

            # gm = sklearn.mixture.GaussianMixture(2, max_iter=20).fit(points[0,...])
            # std = np.diff(gm.means_, axis=0)
            # feats = pd.DataFrame(std) # TODO
            # Append everything to dataframes
            c1 = pd.concat([pd.DataFrame(feats), pd.Series(labels)], axis=1)
            #c1 = pd.DataFrame(feats)
            MLP_profiles = pd.concat([MLP_profiles, c1])
#     pca = PCA(n_components=2)
#     principalComponents = pca.fit_transform(MLP_profiles.T)
#     PC1_loadings = pd.DataFrame(principalComponents[:, 0])
#     plate_loadings = pd.concat([plate_loadings, PC1_loadings], axis=1)
#     continue

#     ALLCELLS.append(NRCELLS)
#     continue
# plt.figure(figsize=(10, 8))
# for i, hist in enumerate(ALLCELLS):
#     plt.hist(hist, bins=100, alpha=0.3, label=PLATES[i].split('_')[-2])
# plt.legend()
# plt.show()

# #%% TODO
# plt.figure(figsize=(15, 15), dpi=300)
# correlations = plate_loadings.corr()
# correlations_array = np.asarray(correlations)
#
# row_linkage = hierarchy.linkage(
#     distance.pdist(correlations_array), method='average', optimal_ordering=True)
# platenames = [x.split('_')[1] for x in PLATES]
# sns.clustermap(correlations, row_linkage=row_linkage, col_linkage=row_linkage, annot=True,
#                xticklabels=platenames, yticklabels=platenames)
# plt.show()
#%% TODO

#%%
    # %% Rename columns and normalize features
    # Rename feature columns
    MLP_profiles.columns = [f"f{x}" for x in range(MLP_profiles.shape[1] - 1)] + ['Metadata_labels']
    print('MLP_profiles shape: ', MLP_profiles.shape)

    # %% Save all the dataframes to .csv files!
    if save_features_to_csv:
        MLP_profiles.to_csv(f'outputs/FinalProfiles/{dataset_name}/MLP_profiles_{platestring}.csv', index=False)

    #from sys import exit
    #exit()

    # %% Analyze feature distributions
    # for df in [plate1df, plate2df, plate3df, plate4df]:
    if evaluate_point_distributions:
        nrRows = 16
        df_MLP = MLP_profiles.iloc[:, :-1]  # Only pass the features

        utils.featureCorrelation(df_MLP, nrRows)
        utils.compoundCorrelation(df_MLP, nrRows)
        utils.createUmap(MLP_profiles, 30)  # need the labels for Umap

    if mAPcalc:
        split = int(MLP_profiles.shape[0] * 0.8)
        MLP_profiles['Metadata_pert_iname'] = list(metadata[metadata['control_type']!='negcon']['pert_iname'])
        MLP_profiles['Metadata_moa'] = list(metadata[metadata['control_type'] != 'negcon']['moa'])

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

    ##% Calculate Percent Replicating on training set used
    if PRcalc:
        print('Calculating Percent Replicating')

        save_name = f"{dataset_name}_{platestring}"  # "TVsplit_allWells_gene_nR3"  ||  "TVsplit_OnlyControls_well_nR3"
        group_by_feature = 'Metadata_labels'

        n_replicates = [4]
        n_samples = 10000

        dataframes = [MLP_profiles]

        descriptions = [platestring]
        print('nReplicates used: ', n_replicates)

        corr_replicating_df = pd.DataFrame()
        for plates, nR, desc in zip(dataframes, n_replicates, descriptions):
            temp_df = CalculatePercentReplicating(plates, group_by_feature, nR, n_samples, desc, percent_matching)
            corr_replicating_df = pd.concat([corr_replicating_df, temp_df], ignore_index=True)

        print(corr_replicating_df[['Description', 'Percent_Replicating']].to_markdown(index=False))

        utils_benchmark.distribution_plot(df=corr_replicating_df, output_file=f"{save_name}_PR.png",
                                          metric="Percent Replicating")

        corr_replicating_df['Percent_Replicating'] = corr_replicating_df['Percent_Replicating'].astype(float)

        plot_corr_replicating_df = (
            corr_replicating_df.rename(columns={'Modality': 'Perturbation'})
                .drop(columns=['Null_Replicating', 'Value_95', 'Replicating'])
        )
        PR = corr_replicating_df['Percent_Replicating'].astype(float)
    else:
        PR = 0
    ##%
    if percent_matching:
        AllResultsDF = pd.concat([AllResultsDF, pd.DataFrame({'mAP model': AP.iloc[:, 1].mean(),
                                                              'PR model': [PR]
                                                              })])
        sorted_dictionary = {k: [v] for k, v in sorted(average_perturbation_map.items(), key=lambda item: item[1], reverse=True)}
        bestMOAs = pd.concat([bestMOAs, pd.DataFrame(sorted_dictionary)])
    else:
        AllResultsDF = pd.concat([AllResultsDF, pd.DataFrame({'Training mAP model': AP.iloc[:split, 1].mean(),
                                                              'Validation mAP model': AP.iloc[split:, 1].mean(),
                                                              'PR model': [PR]
                                                              })])


PLATES = [x.split('_')[-2] for x in PLATES]
AllResultsDF['plate'] = PLATES

# Load BM results
if percent_matching:
    BMdf = pd.read_csv(f'/Users/rdijk/Documents/ProjectFA/Benchmarks/{dataset_name}_BM_MOA.csv')
else:
    BMdf = pd.read_csv(f'/Users/rdijk/Documents/ProjectFA/Benchmarks/{dataset_name}_BM.csv')
AllResultsDF = AllResultsDF.merge(BMdf, on='plate')

AllResultsDF = AllResultsDF.set_index('plate')
cols = AllResultsDF.columns.tolist()
if percent_matching:
    if PRcalc:
        cols = [cols[0], cols[2], cols[1], cols[3]]
    else:
        cols = [cols[0], cols[2]]
else:
    if PRcalc:
        if dataset_name == 'Stain5':
            cols = [cols[0], cols[3], cols[1], cols[4], cols[2], cols[5], cols[6]]
        else:
            cols = [cols[0], cols[3], cols[1], cols[4], cols[2], cols[5]]
    else:
        if dataset_name == 'Stain5':
            cols = [cols[0], cols[3], cols[1], cols[4], cols[2], cols[5]]
        else:
            cols = [cols[0], cols[3], cols[1], cols[4]]
AllResultsDF = AllResultsDF[cols]
print(AllResultsDF.round(2).to_markdown())

if percent_matching:
    AllResultsDF.to_csv(f'/Users/rdijk/Documents/ProjectFA/FinalModelResults/FinalModelResults_{dataset_name}.csv')
else:
    AllResultsDF.to_csv(f'/Users/rdijk/Documents/ProjectFA/FinalModelResults/FinalModelResults_replicating_{dataset_name}.csv')

if percent_matching:
    bestMOAs.to_csv(f'/Users/rdijk/Documents/ProjectFA/FinalModelResults/bestMOAs_{dataset_name}.csv')
