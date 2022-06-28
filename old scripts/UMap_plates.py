import pandas as pd
import os
import matplotlib.pyplot as plt
import umap.plot
from pycytominer.operations.transform import RobustMAD
from pycytominer import feature_select

dataset_name = 'Stain2'

rootDir_MLP = fr'/Users/rdijk/PycharmProjects/featureAggregation/outputs/{dataset_name}'
rootdir_BM = fr'/Users/rdijk/Documents/Data/ProcessedData/{dataset_name}/profiles'
MLP_platedirs = [x[2] for x in os.walk(rootDir_MLP)][0][1:]
MLP_platenames = [x.split('_')[-1] for x in MLP_platedirs]

BM_platedirs = [x[2] for x in os.walk(rootdir_BM)][0]
BM_platedirs = [x for x in BM_platedirs if x in MLP_platenames]

MLP_platedirs.sort()
BM_platedirs.sort()

commonFeatureNames = pd.read_csv('/Users/rdijk/Documents/Data/RawData/CommonFeatureNames.csv', index_col=False)
commonFeatureNames = [x.split('.')[1] for x in commonFeatureNames.iloc[:, 0]]
commonFeatureNames.append('label')
commonFeatureNames.append('pert_iname')

rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/Stain2'
metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv',
                       index_col=False)
metadata = metadata[metadata.control_type != 'negcon'].reset_index()

MLPdfs = []
BMdfs = []
i = 0
negconsStain = ["A11", "B18", "D17", "D19", "E07", "E08", "F07", "F24", "G20", "G23", "H10", "I12", "I13", "J01", "J21",
                "K24", "M05", "M06", "N06", "N22", "O14", "O19", "P02", "P11"]

for MLP, BM in zip(MLP_platedirs, BM_platedirs):
    lblsMLP = [MLP_platenames[i][:-4]] * 360
    lblsBM = [MLP_platenames[i][:-4]] * 384

    temp1 = pd.concat([pd.read_csv(os.path.join(rootDir_MLP, MLP), index_col=False), pd.Series(lblsMLP, name='label')], axis=1)
    temp1['pert_iname'] = metadata['pert_iname']

    # sort compounds by training/validation compound and split data
    #temp1 = temp1.sort_values(by='pert_iname').iloc[:288, :]
    temp1 = temp1.sort_values(by='pert_iname').iloc[288:, :]

    MLPdfs.append(temp1)

    temp2 = pd.concat([pd.read_csv(os.path.join(rootdir_BM, BM), index_col=False), pd.Series(lblsBM, name='label')], axis=1)
    features = temp2[commonFeatureNames[:-2]]
    scaler = RobustMAD()
    fitted_scaler = scaler.fit(features)
    features = fitted_scaler.transform(features)
    # features = features.dropna(axis=1, how='any')
    # features = feature_select(features, operation=["variance_threshold", "correlation_threshold",
    #                                                 "drop_na_columns", "blocklist"])

    features['label'] = temp2['label']
    features = features[~temp2.Metadata_Well.isin(negconsStain)].reset_index(drop=True) # remove negcons
    features['pert_iname'] = metadata['pert_iname']

    #features = features.sort_values(by='pert_iname').iloc[:288, :]
    features = features.sort_values(by='pert_iname').iloc[288:, :]

    BMdfs.append(features)
    i += 1


# %% Create UMap
plt.figure(figsize=(14, 14), dpi=400)
BMDF = pd.concat([x for x in BMdfs])
featuresBM = BMDF.iloc[:, :-2]
labels = BMDF.iloc[:, -2]
reducer = umap.UMAP(random_state=42)
embedding1 = reducer.fit(featuresBM)
umap.plot.points(embedding1, labels=labels, color_key_cmap='Paired', show_legend=True)
plt.gca().set_aspect('equal', 'datalim')
plt.title(f'UMAP BM {dataset_name}')
plt.show()

plt.figure(figsize=(14, 14), dpi=400)
MLPDF = pd.concat([x for x in MLPdfs])
featuresMLP = MLPDF.iloc[:, :-3]
labels = MLPDF.iloc[:, -2]
reducer = umap.UMAP(random_state=42)
embedding2 = reducer.fit(featuresMLP)
umap.plot.points(embedding2, labels=labels, color_key_cmap='Paired', show_legend=True)
plt.gca().set_aspect('equal', 'datalim')
plt.title(f'UMAP MLP {dataset_name}')
plt.show()


plt.figure(figsize=(14, 14), dpi=400)
labels = BMDF.iloc[:, -1]
umap.plot.points(embedding1, labels=labels, color_key_cmap='Paired', show_legend=True)
plt.gca().set_aspect('equal', 'datalim')
plt.title(f'UMAP BM {dataset_name}')
plt.show()

plt.figure(figsize=(14, 14), dpi=400)
labels = MLPDF.iloc[:, -1]
umap.plot.points(embedding2, labels=labels, color_key_cmap='Paired', show_legend=True)
plt.gca().set_aspect('equal', 'datalim')
plt.title(f'UMAP MLP {dataset_name}')
plt.show()
