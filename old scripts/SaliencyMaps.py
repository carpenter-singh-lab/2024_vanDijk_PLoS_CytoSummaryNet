"""
This script is old, please look at the jupyter notebooks for up to date saliency visualizations.
"""
## Standard libraries
import os
import pandas as pd
from tabulate import tabulate

## Seeds
import random
import numpy as np

## PyTorch
import torch
import torch.utils.data as data

# Custom libraries
from networks.SimpleMLPs import MLPsumV2
from dataloader_pickles import DataloaderEvalV5
import utils
from pytorch_metric_learning import losses, distances

## UMAP libraries
import matplotlib.pyplot as plt
import umap.plot
import matplotlib as mpl
import copy

NUM_WORKERS = 0
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Set random seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# %%
# Load model
save_name_extension = 'general_ckpt_simpleMLP_V1'
model_name = save_name_extension
print('Loading:', model_name)

input_dim = 1324
kFilters = 1 / 2
latent_dim = 2048
output_dim = 2048
model = MLPsumV2(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim,
                 k=kFilters, dropout=0, cell_layers=1,
                 proj_layers=2, reduction='sum')
if torch.cuda.is_available():
    model.cuda()

path = r'../wandb/latest-run/files'
#path = r'wandb/run-20220427_191823-och8e6jv/files'  # Stain3+Stain4
# path = r'wandb/offline-run-20220426_090449-1rwtheol/files' # best Stain4
# path = r'wandb/run-20220411_202347-3i0drmgj/files' # best Stain2

models = os.listdir(path)
fullpath = os.path.join(path, model_name)
if 'ckpt' in model_name:
    model.load_state_dict(torch.load(fullpath)['model_state_dict'])
else:
    model.load_state_dict(torch.load(fullpath))
model.eval()

# %% Select specific plate and compound
plates = ['BR00112204_FS']
compound = None  # 'Compound2' # if select random: None

calc_and_store_saliencies = True
saliencies = torch.tensor([]).requires_grad_()
cell_nrs = []
well_positions = []

NR_WELLS_SHOWN = 10  # In saliency plot

# %%
# Load all data
rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/Stain2'
metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv',
                       index_col=False)
plateDirs = [x[0] for x in os.walk(rootDir)][1:]

plateDirs = [x for x in plateDirs if any(substr in x for substr in plates)]

platestring = plateDirs[0].split('_')[-2]
print('Calculating results for: ' + platestring)
metadata = utils.addDataPathsToMetadata(rootDir, metadata, plateDirs)

# if compound is not None:
#     compound_well_path = pd.DataFrame(metadata[metadata['pert_iname'] == compound]).iloc[:1, :]
#     metadata = metadata[metadata['pert_iname'] != compound]
#     metadata = pd.concat([metadata, compound_well_path])

# Filter the data and create numerical labels
df_prep = utils.filterData(metadata, '', encode='pert_iname')
# Add all data to one DF
Total, _ = utils.train_val_split(df_prep, 1.0, sort=False)
valTotal = Total.sort_values(by='Metadata_labels').iloc[288:, :].reset_index(drop=True)

valset = DataloaderEvalV5(valTotal)
loader = data.DataLoader(valset, batch_size=1, shuffle=False,  # 96
                         drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)
# %%
F = []
L = []
for points, label, well_position in loader:
    # points = points[:, torch.randperm(points.shape[1]), :] # random shuffle for testing
    points.requires_grad_()
    features, _ = model(points)
    F.append(features)
    L.append(label)

    if calc_and_store_saliencies:
        saliencies = torch.cat([saliencies, points], dim=1)
        cell_nrs.append(points.shape[1])
        well_positions.append([well_position]*points.shape[1])
# %%
print('Calculating Salient Features')
loss = losses.SupConLoss(distance=distances.CosineSimilarity())

features_loss = loss(torch.cat(F), torch.cat(L))
features_loss.backward()

if calc_and_store_saliencies:
    saliencies = torch.sum(saliencies.grad.data.abs(), dim=2)
    a = pd.DataFrame({'well': well_positions,
                      'saliencies': saliencies})

# Retrieve the saliency map and also pick the mean or maximum value from all cells for a given feature.
saliency_feat = torch.sum(points.grad.data.abs(), dim=1)
saliency_cells = torch.sum(points.grad.data.abs(), dim=2)

# Return top X features
top_idx = np.argpartition(saliency_feat.numpy(), 20)[0]

# Get Feature Names
df = pd.read_csv('/Users/rdijk/Documents/Data/RawData/CommonFeatureNames.csv', index_col=False)

newDF = pd.DataFrame()
newDF['FeatureNames'] = df.iloc[top_idx]
newDF['Saliency'] = saliency_feat[0, top_idx]
newDF = newDF.sort_values(by='Saliency', ascending=False)

df_prep['plate1'] = df_prep['plate1'].str.split('/', expand=True).iloc[:, -2].str.split('_', expand=True).iloc[:, -2]
clabel = label.item()
well_inf = df_prep[['pert_iname', 'well_position', 'plate1']][df_prep.Metadata_labels == clabel].head(1)
print(well_inf)
print('Number of cells:', points.shape[1])
print(tabulate(newDF.iloc[:20, :], headers='keys', tablefmt='github', showindex=False))

# %% Plot cell saliency in UMAP plot
themes = ['fire', 'viridis', 'inferno', 'darkblue', 'darkred', 'darkgreen']
point_list = [23, 35, 39, 43]  # sirolimus, skepinone-l, valrubicin
reducer = umap.UMAP(random_state=42, metric='euclidean', )
embedding = reducer.fit(points[point_list, ...].detach().numpy().reshape(2500 * len(point_list), 1324))

fig, ax = plt.subplots(1, 1, dpi=300)
for idx, i, theme in zip(range(len(themes)), point_list, themes):
    c_embedding = copy.deepcopy(embedding)
    c_embedding.embedding_ = embedding.embedding_[2500 * idx:2500 * (idx + 1), ...]
    values = saliency_cells[i, :].detach().numpy()

    umap.plot.points(c_embedding, values=values, theme=theme, ax=ax)
    # psm = plt.pcolormesh([values, values], cmap=mpl.cm.get_cmap(theme))
    # cb = plt.colorbar(psm, ax=ax)

# plt.title(f'Cell saliency\n {well_inf}')
plt.show()
