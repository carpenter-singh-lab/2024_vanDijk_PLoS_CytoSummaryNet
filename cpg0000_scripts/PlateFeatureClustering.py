## Standard libraries
import os
import glob
import pandas as pd

## Seeds
import random
import numpy as np

## PyTorch
import torch

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Plate clusters
from scipy.spatial import distance
from scipy.cluster import hierarchy

import utils
import copy

NUM_WORKERS = 0
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# Set random seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# %% Load data
rootdir = '/Users/rdijk/Documents/Data/ProcessedData/Stain2/profiles'
plateDirs = glob.glob(os.path.join(rootdir, '*.csv'))
plateDirs.remove(os.path.join(rootdir, 'BR00112200.csv')) # Was already removed
plateDirs.remove(os.path.join(rootdir, 'BR00112203.csv')) # Remove 3
# plateDirs.remove(os.path.join(rootdir, 'BR00113819.csv')) # Remove 4
# plateDirs.remove(os.path.join(rootdir, 'BR00113820.csv')) # Remove 4
# plateDirs.remove(os.path.join(rootdir, 'BR00113821.csv')) # Remove 4

rootdir2 = '/Users/rdijk/Documents/Data/ProcessedData/Stain3/profiles'
plateDirs2 = glob.glob(os.path.join(rootdir2, '*.csv'))
plateDirs2.remove(os.path.join(rootdir2, 'BR00115130.csv')) # Remove 3
# plateDirs2.remove(os.path.join(rootdir2, 'BR00115132.csv')) # Remove 4
# plateDirs2.remove(os.path.join(rootdir2, 'BR00115132highexp.csv')) # Remove 4
# plateDirs2.remove(os.path.join(rootdir2, 'BR00115134multiplane.csv')) # Remove 4
# plateDirs2.remove(os.path.join(rootdir2, 'BR00115134bin1.csv')) # Remove 4
# plateDirs2.remove(os.path.join(rootdir2, 'BR00115126.csv')) # Remove 4
# plateDirs2.remove(os.path.join(rootdir2, 'BR00115126highexp.csv')) # Remove 4

rootdir3 = '/Users/rdijk/Documents/Data/ProcessedData/Stain4/profiles'
plateDirs3 = glob.glob(os.path.join(rootdir3, '*.csv'))
# plateDirs3.remove(os.path.join(rootdir3, 'BR00116634bin1.csv')) # Remove 3
# plateDirs3.remove(os.path.join(rootdir3, '200921_193743-V.csv')) # Remove 4
# plateDirs3.remove(os.path.join(rootdir3, '200921_193743-Vhighexp.csv')) # Remove 4
# plateDirs3.remove(os.path.join(rootdir3, '200922_044247-V.csv')) # Remove 4
# plateDirs3.remove(os.path.join(rootdir3, 'BR00116633bin1.csv')) # Remove 4


# plateDirs3.remove(os.path.join(rootdir3, 'BR00116624highexp.csv'))
# plateDirs3.remove(os.path.join(rootdir3, 'BR00116624bin1.csv'))
# plateDirs3.remove(os.path.join(rootdir3, 'BR00116620bin1.csv'))
# plateDirs3.remove(os.path.join(rootdir3, 'BR00116620highexp.csv'))
# plateDirs3.remove(os.path.join(rootdir3, 'BR00116621highexp.csv'))
# plateDirs3.remove(os.path.join(rootdir3, 'BR00116621bin1.csv'))
# plateDirs3.remove(os.path.join(rootdir3, 'BR00116634highexp.csv'))
# plateDirs3.remove(os.path.join(rootdir3, 'BR00116633highexp.csv'))
# plateDirs3.remove(os.path.join(rootdir3, 'BR00116632highexp.csv'))
# plateDirs3.remove(os.path.join(rootdir3, 'BR00116622.csv'))
# plateDirs3.remove(os.path.join(rootdir3, '200922_015124-V.csv'))
# plateDirs3.remove(os.path.join(rootdir3, '200922_015124-Vhighexp.csv'))
# plateDirs3.remove(os.path.join(rootdir3, 'BR00116622highexp.csv'))



rootdir4 = '/Users/rdijk/Documents/Data/ProcessedData/Stain5/profiles'
plateDirs4 = glob.glob(os.path.join(rootdir4, '*.csv'))
plateDirs4.remove(os.path.join(rootdir4, 'BR00120269confocal.csv')) # Remove 2
plateDirs4.remove(os.path.join(rootdir4, 'BR00120267confocal.csv')) # Remove 2
plateDirs4.remove(os.path.join(rootdir4, 'BR00120269.csv')) # Remove 3
plateDirs4.remove(os.path.join(rootdir4, 'BR00120267.csv')) # Remove 3

rootdir5 = '/Users/rdijk/Documents/Data/ProcessedData/Stain5/profiles/CondA'
plateDirs5 = glob.glob(os.path.join(rootdir5, '*.csv'))
plateDirs5.remove(os.path.join(rootdir5, 'BR00120547.csv')) # Remove 3
plateDirs5.remove(os.path.join(rootdir5, 'BR00120280.csv')) # Remove 3
plateDirs5.remove(os.path.join(rootdir5, 'BR00120278.csv')) # Remove 3
plateDirs5.remove(os.path.join(rootdir5, 'BR00120279.csv')) # Remove 3

plateDirs = plateDirs + plateDirs2 + plateDirs3 + plateDirs4 + plateDirs5
#plateDirs = plateDirs2
#plateDirs = plateDirs + plateDirs4 + plateDirs5

# Select subselection
# plates = ['BR00115134.csv', 'BR00115125.csv', 'BR00115133highexp.csv', 'BR00112201.csv', 'BR00112198.csv', 'BR00112204.csv']
# plates = ['BR00115126.csv', 'BR00115134.csv', 'BR00115125.csv', 'BR00115133highexp.csv', 'BR00116625highexp.csv', 'BR00116628highexp.csv', 'BR00116629highexp.csv']
# plateDirs = [x for x in plateDirs if any(substr in x for substr in plates)]
# plateDirs = plateDirs3

# plateDirs = ['/Users/rdijk/PycharmProjects/featureAggregation/outputs/Stain2_wellAugment_noCP/MLP_profiles_BR00112198.csv',
# '/Users/rdijk/PycharmProjects/featureAggregation/outputs//Stain2_wellAugment_noCP/MLP_profiles_BR00112204.csv',
# '/Users/rdijk/PycharmProjects/featureAggregation/outputs//Stain2_wellAugment_noCP/MLP_profiles_BR00112201.csv',
# '/Users/rdijk/PycharmProjects/featureAggregation/outputs//Stain2_wellAugment_noCP/MLP_profiles_BR00112197repeat.csv',
#  '/Users/rdijk/PycharmProjects/featureAggregation/outputs//Stain2_wellAugment_noCP/MLP_profiles_BR00112202.csv',
# '/Users/rdijk/PycharmProjects/featureAggregation/outputs//Stain2_wellAugment_noCP/MLP_profiles_BR00112197binned.csv',
#  '/Users/rdijk/PycharmProjects/featureAggregation/outputs//Stain2_wellAugment_noCP/MLP_profiles_BR00112197standard.csv']
# plateDirs = ['/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00113818.csv',
#  '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00113819.csv',
#  '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00113821.csv',
#  '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00113820.csv',
# '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00112198.csv',
# '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00112204.csv',
# '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00112199.csv',
# '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00112201.csv',
# '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00112197repeat.csv',
# '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00112203.csv',
#  '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00112202.csv',
# '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00112197binned.csv',
#  '/Users/rdijk/PycharmProjects/featureAggregation/outputs/MLP_profiles_BR00112197standard.csv']


metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv',
                       index_col=False)

metadata = metadata.sort_values(by='pert_iname')

commonFeatureNames = pd.read_csv('/Users/rdijk/Documents/Data/RawData/CommonFeatureNames.csv', index_col=False)
commonFeatureNames = [x.split('.')[1] for x in commonFeatureNames.iloc[:, 0]]

# %% Calculate Percent Replicating on training set used
plate_loadings = pd.DataFrame()
for filepath in plateDirs:
    platestring = filepath.split('/')[-1]

    df = pd.read_csv(filepath, index_col=False, low_memory=False)
    features = df.iloc[:, :-1]

    # Sort according to pert_iname
    df = df.set_index('Metadata_Well')
    df = df.reindex(index=metadata['well_position'])
    df = df.reset_index()

    # Select 1324 features
    features = df[commonFeatureNames]

    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)

    features = features[~np.isnan(features).any(axis=1)] # drop nan rows


    ### PCA ANALYSIS
    # df = df.sort_values(by='Metadata_labels')
    # df_group = df.groupby('Metadata_labels')
    # for name, group in df_group:
    #     pca = PCA(n_components=2)
    #     principalComponents = pca.fit_transform(group.T)
    #     PC1_loadings = pd.DataFrame(principalComponents[:, 0])
    #     plate_loadings = pd.concat([plate_loadings, PC1_loadings], axis=1)
    # print('done')
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(features.T)
    PC1_loadings = pd.DataFrame(principalComponents[:, 0]) # 7/8
    PC1_loadings = (PC1_loadings - PC1_loadings.min()) / (PC1_loadings.max() - PC1_loadings.min())
    PC2_loadings = pd.DataFrame(principalComponents[:, 1])
    PC3_loadings = pd.DataFrame(principalComponents[:, 2])
    PC_loadings = pd.DataFrame(principalComponents.sum(1))
    plate_loadings = pd.concat([plate_loadings, PC1_loadings], axis=1)


plt.figure(figsize=(16, 16), dpi=500)
#platenames = [x.split('/')[-1].split('.')[0].split('_')[-1] for x in plateDirs]
platenames = [x.split('/')[-3] for x in plateDirs[:-len(plateDirs5)]]
#platenames = platenames + [x.split('/')[-2] for x in plateDirs[-len(plateDirs5):]]
correlations = plate_loadings.corr()
correlations_array = np.asarray(correlations)

row_linkage = hierarchy.linkage(
    distance.pdist(correlations_array), method='average', optimal_ordering=True)

sns.clustermap(correlations, row_linkage=row_linkage, col_linkage=row_linkage, annot=False,
               xticklabels=platenames, yticklabels=platenames, figsize=(16,16))

#sns.heatmap(correlations, xticklabels=platenames, yticklabels=platenames, annot=False)
#c = sns.clustermap(Var_Corr, xticklabels=platenames, yticklabels=platenames, annot=False, metric='correlation')
#plt.title('PC1 loadings correlation per plate')

#%% Calculate clusters
cluster_numbers = hierarchy.fcluster(row_linkage, 2, criterion='maxclust')
a = pd.DataFrame({'plate': platenames, 'cluster': cluster_numbers})
a = a.groupby('cluster')
for key, item in a:
    print(a.get_group(key).to_markdown(), "\n\n")

