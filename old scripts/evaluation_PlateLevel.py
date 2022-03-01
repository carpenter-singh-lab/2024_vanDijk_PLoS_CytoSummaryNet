## Standard libraries
import os
from tqdm import tqdm
import pandas as pd

## Seeds
import random
import numpy as np

## PyTorch
import torch
import torch.utils.data as data

# Custom libraries
from networks.SimpleMLPs import MLP
from dataloader_pickles import DataloaderEvalV4
from utils import CalculatePercentReplicating
import utils
import utils_benchmark
from pycytominer.operations.transform import RobustMAD

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

# %% Load model
save_name_extension = 'general_ckpt_simpleMLP_V1'  # extension of the saved model
model_name = save_name_extension
print('Loading:', model_name)

BS = 64  # batch size
nr_cells = 600  # nr of cells sampled from each well (no more than 1200 found in compound plates)
input_dim = 838  # 1938 Cells // 838 FS
kFilters = 4  # times DIVISION of filters in model
latent_dim = 1028
output_dim = 512
model = MLP(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim, k=kFilters)

save_features_to_csv = False
evaluate_point_distributions = False
path = r'../wandb/run-20220216_120614-srr5qywd/files'

models = os.listdir(path)
fullpath = os.path.join(path, model_name)
if 'ckpt' in model_name:
    model.load_state_dict(torch.load(fullpath)['model_state_dict'])
else:
    model.load_state_dict(torch.load(fullpath))
model.eval()
# %% Load all data
rootDir = r'/Users/rdijk/PycharmProjects/featureAggregation/datasets/CPJUMP1'

# Set paths for all training/validation files
plateDirTrain1 = 'DataLoader_BR00117010_unfiltered'
tdir1 = os.path.join(rootDir, plateDirTrain1)
plateDirTrain2 = 'DataLoader_BR00117011_unfiltered'
tdir2 = os.path.join(rootDir, plateDirTrain2)
plateDirVal1 = 'DataLoader_BR00117012_unfiltered'
vdir1 = os.path.join(rootDir, plateDirVal1)
plateDirVal2 = 'DataLoader_BR00117013_unfiltered'
vdir2 = os.path.join(rootDir, plateDirVal2)

# Load csv for pair formation
metadata = pd.read_csv('/Users/rdijk/Documents/Data/RawData/CPJUMP1_compounds/JUMP_target_compound_metadata_wells.csv', index_col=False)

# Load all absolute paths to the pickle files for training
filenamesTrain1 = [os.path.join(tdir1, file) for file in os.listdir(tdir1)]
filenamesTrain2 = [os.path.join(tdir2, file) for file in os.listdir(tdir2)]
filenamesTrain1.sort()
filenamesTrain2.sort()
# and for validation
filenamesVal1 = [os.path.join(vdir1, file) for file in os.listdir(vdir1)]
filenamesVal1.sort()
filenamesVal2 = [os.path.join(vdir2, file) for file in os.listdir(vdir2)]
filenamesVal2.sort()

# Create preprocessing dataframe for both validation and training
metadata['plate1'] = filenamesTrain1
metadata['plate2'] = filenamesTrain2
metadata['plate3'] = filenamesVal1
metadata['plate4'] = filenamesVal2

# Filter the data and create numerical labels
df_prep = utils.filterData(metadata, 'negcon', encode='pert_iname', sort=False)
# Add all data to one DF
Total, _ = utils.train_val_split(df_prep, 1, sort=False)

trainset = DataloaderEvalV4(Total, nr_cells=nr_cells, multi_sample=False)

train_loader = data.DataLoader(trainset, batch_size=BS, shuffle=False, collate_fn=utils.my_collate_eval,
                               drop_last=False, pin_memory=False, num_workers=NUM_WORKERS)

# %% Calculate NCEloss + Create feature dataframes
MLP_profiles = pd.DataFrame()
BM_profiles = pd.DataFrame()

print('Calculating Features')
with torch.no_grad():
    for idx, (points, labels, aggregated_points, agg_target) in enumerate(tqdm(train_loader)):
        points = points.to(device)
        feats, _ = model(points)
        # Append everything to dataframes
        c1 = pd.concat([pd.DataFrame(feats), pd.Series(labels)], axis=1)
        MLP_profiles = pd.concat([MLP_profiles, c1])

        d1 = pd.concat([pd.DataFrame(aggregated_points), pd.Series(agg_target)], axis=1)
        BM_profiles = pd.concat([BM_profiles, d1])

# %% Rename columns and normalize features
# Rename feature columns
MLP_profiles.columns = [f"f{x}" for x in range(MLP_profiles.shape[1] - 1)] + ['Metadata_labels']
BM_profiles.columns = [f"f{x}" for x in range(BM_profiles.shape[1] - 1)] + ['Metadata_labels']

print('MLP_profiles shape: ', MLP_profiles.shape)
print('BM_profiles shape: ', BM_profiles.shape)

# Robust MAD normalize features per plate
MLP_profiles_norm = pd.DataFrame()
unit = len(df_prep)
for i in range(4):
    scaler = RobustMAD()
    if i == 3:
        MLP_profiles.iloc[int(3 * unit):, :]
    cplate = MLP_profiles.iloc[int(i*unit):int((i+1)*unit), :-1]
    fitted_scaler = scaler.fit(cplate)
    profiles_norm = fitted_scaler.transform(cplate)
    MLP_profiles_norm = pd.concat([MLP_profiles_norm, profiles_norm])
    # profiles_norm['Metadata_labels'] = MLP_profiles.iloc[int(i*unit):int((i+1)*unit), -1]
    # profiles_norm.to_csv(f'/Users/rdijk/Documents/Data/profiles/2020_11_04_CPJUMP1/BR0011701{i}_MLP.csv', index=False)

BM_profiles_norm = pd.DataFrame()
for i in range(4):
    scaler = RobustMAD()
    if i == 3:
        BM_profiles.iloc[int(3 * unit):, :]
    cplate = BM_profiles.iloc[int(i*unit):int((i+1)*unit), :-1]
    fitted_scaler = scaler.fit(cplate)
    profiles_norm = fitted_scaler.transform(cplate)
    BM_profiles_norm = pd.concat([BM_profiles_norm, profiles_norm])
    # profiles_norm['Metadata_labels'] = BM_profiles.iloc[int(i * unit):int((i + 1) * unit), -1]
    # profiles_norm.to_csv(f'/Users/rdijk/Documents/Data/profiles/2020_11_04_CPJUMP1/BR0011701{i}_BM.csv', index=False)

MLP_profiles.iloc[:, :-1] = MLP_profiles_norm
BM_profiles.iloc[:, :-1] = BM_profiles_norm

# %% Save all the dataframes to .csv files!
if save_features_to_csv:
    MLP_profiles.to_csv(r'outputs/MLP_profiles.csv', index=False)
    BM_profiles.to_csv(r'outputs/BM_profiles.csv', index=False)

# %% Analyze feature distributions
# for df in [plate1df, plate2df, plate3df, plate4df]:
if evaluate_point_distributions:
    nrRows = 16
    df_MLP = MLP_profiles.iloc[:, :-1]  # Only pass the features
    df_BM = BM_profiles.iloc[:, :-1]

    utils.featureCorrelation(df_MLP, nrRows)
    utils.featureCorrelation(df_BM, nrRows)

    utils.compoundCorrelation(df_MLP, nrRows)
    utils.compoundCorrelation(df_BM, nrRows)

    # UMAP
    utils.createUmap(MLP_profiles, 30)  # need the labels for Umap
    utils.createUmap(BM_profiles, 30)


# %% Calculate Percent Replicating on training set used
print('Calculating Percent Replicating')

save_name = "Generalized"  # "TVsplit_allWells_gene_nR3"  ||  "TVsplit_OnlyControls_well_nR3"
group_by_feature = 'Metadata_labels'

n_replicatesT = int(round(MLP_profiles['Metadata_labels'].value_counts().mean()))
n_samples = 10000

dataframes = [MLP_profiles, BM_profiles]
#dataframes = [[MLP_profiles.iloc[:320, :], MLP_profiles.iloc[320:640, :]], [MLP_profiles.iloc[640:960, :], MLP_profiles.iloc[960:, :]]]

nReplicates = [6, 6]
descriptions = ['MLP', 'BM']
print('nReplicates used: ', nReplicates)

corr_replicating_df = pd.DataFrame()
for plates, nR, desc in zip(dataframes, nReplicates, descriptions):
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
