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
from dataloader_pickles import DataloaderTrainV3
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
save_name_extension = 'model_bestval_simpleMLP_V1'  # extension of the saved model
model_name = save_name_extension
print('Loading:', model_name)

BS = 16  # batch size
nr_cells = 600  # nr of cells sampled from each well (no more than 1200 found in compound plates)
input_dim = 1938  # 1938
kFilters = 4  # times DIVISION of filters in model
latent_dim = 1028
output_dim = 512
model = MLP(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim, k=kFilters)

save_features_to_csv = False
path = r'../wandb/latest-run/files'
models = os.listdir(path)
fullpath = os.path.join(path, model_name)
if 'ckpt' in model_name:
    model.load_state_dict(torch.load(fullpath)['model_state_dict'])
else:
    model.load_state_dict(torch.load(fullpath))
model.eval()
model.double()
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
df = utils.filterData(metadata, 'negcon', encode='pert_iname')

TrainTotal = df.drop(['plate3', 'plate4'], axis=1)
ValTotal = df.drop(['plate1', 'plate2'], axis=1)

trainset = DataloaderTrainV3(TrainTotal, nr_cells=nr_cells)
valset = DataloaderTrainV3(ValTotal, nr_cells=nr_cells)

train_loader = data.DataLoader(trainset, batch_size=BS, shuffle=False,
                               drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
val_loader = data.DataLoader(valset, batch_size=BS, shuffle=False,
                             drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
# %% Calculate NCEloss + Create feature dataframes
plate1df = pd.DataFrame()
plate2df = pd.DataFrame()
plate3df = pd.DataFrame()
plate4df = pd.DataFrame()

print('Calculating Features')
with torch.no_grad():
    for idx, (points, labels) in enumerate(tqdm(train_loader)):
        points, labels = torch.cat([x for x in points]), torch.cat([x for x in labels])
        points = points.to(device)
        feats, _ = model(points)
        # Append everything to dataframes
        plate1df = pd.concat([plate1df, pd.DataFrame(feats[:points.shape[0] // 2, :])])
        plate2df = pd.concat([plate2df, pd.DataFrame(feats[points.shape[0] // 2:, :])])

# Validation
with torch.no_grad():
    for i, (points, labels) in enumerate(tqdm(val_loader)):
        points, labels = torch.cat([x for x in points]), torch.cat([x for x in labels])
        points = points.to(device)
        feats, _ = model(points)
        # Append everything to dataframes
        plate3df = pd.concat([plate3df, pd.DataFrame(feats[:points.shape[0] // 2])])
        plate4df = pd.concat([plate4df, pd.DataFrame(feats[points.shape[0] // 2:])])

# %% Create .csv to calculate Percent Replicating/Matching
""" 
1. Specify .csv profiles that you want to compare the trained model to. 
2. Load the .csv files and remove all of the cell features. 
3. Add the cell features which were created by the model to the .csv and save with a new extension. 
Then use these .csv files in a different python script to calculate your metrics. 
"""
print('Creating new .csv feature files')
# Training
Tplate1 = pd.read_csv('/Users/rdijk/Documents/Data/profiles/Training/BR00117010/BR00117010_normalized.csv')
Tplate2 = pd.read_csv('/Users/rdijk/Documents/Data/profiles/Training/BR00117011/BR00117011_normalized.csv')

# Validation
Vplate1 = pd.read_csv('/Users/rdijk/Documents/Data/profiles/Validation/BR00117012/BR00117012_normalized.csv')
Vplate2 = pd.read_csv('/Users/rdijk/Documents/Data/profiles/Validation/BR00117013/BR00117013_normalized.csv')

# Remove all cell features
Tplate1.drop(Tplate1.columns[11:], axis=1, inplace=True)
Tplate2.drop(Tplate2.columns[11:], axis=1, inplace=True)
Vplate1.drop(Vplate1.columns[11:], axis=1, inplace=True)
Vplate2.drop(Vplate2.columns[11:], axis=1, inplace=True)

# Filter data and add label column
Tplate1 = utils.filterData(Tplate1, 'negcon', encode='Metadata_pert_iname', mode='eval')
Tplate2 = utils.filterData(Tplate2, 'negcon', encode='Metadata_pert_iname', mode='eval')
Vplate1 = utils.filterData(Vplate1, 'negcon', encode='Metadata_pert_iname', mode='eval')
Vplate2 = utils.filterData(Vplate2, 'negcon', encode='Metadata_pert_iname', mode='eval')

# Rename feature columns
plate1df.columns = [f"f{x}" for x in range(plate1df.shape[1])]
plate2df.columns = [f"f{x}" for x in range(plate2df.shape[1])]
plate3df.columns = [f"f{x}" for x in range(plate3df.shape[1])]
plate4df.columns = [f"f{x}" for x in range(plate4df.shape[1])]
print('platedf shape: ', plate1df.shape)

# Robust MAD normalize features
scaler1 = RobustMAD()
scaler2 = RobustMAD()
scaler3 = RobustMAD()
scaler4 = RobustMAD()
fitted_scaler1 = scaler1.fit(plate1df)
fitted_scaler2 = scaler2.fit(plate2df)
fitted_scaler3 = scaler3.fit(plate3df)
fitted_scaler4 = scaler4.fit(plate4df)

plate1df = fitted_scaler1.transform(plate1df)
plate2df = fitted_scaler2.transform(plate2df)
plate3df = fitted_scaler3.transform(plate3df)
plate4df = fitted_scaler4.transform(plate4df)

# Add cell features of the model
Tplate1 = pd.concat([Tplate1.reset_index(drop=True), plate1df.reset_index(drop=True)], axis=1)
Tplate2 = pd.concat([Tplate2.reset_index(drop=True), plate2df.reset_index(drop=True)], axis=1)
Vplate1 = pd.concat([Vplate1.reset_index(drop=True), plate3df.reset_index(drop=True)], axis=1)
Vplate2 = pd.concat([Vplate2.reset_index(drop=True), plate4df.reset_index(drop=True)], axis=1)

# %% Save all the dataframes to .csv files!
if save_features_to_csv:
    Tplate1.to_csv(r'outputs/MLP_profiles_plate1.csv', index=False)
    Tplate2.to_csv(r'outputs/MLP_profiles_plate2.csv', index=False)
    Vplate1.to_csv(r'outputs/MLP_profiles_plate3.csv', index=False)
    Vplate2.to_csv(r'outputs/MLP_profiles_plate4.csv', index=False)

# %% Analyze feature distributions
# for df in [plate1df, plate2df, plate3df, plate4df]:
df = plate4df
utils.featureCorrelation(df, 16)
utils.compoundCorrelation(df, 16)

df2 = Vplate2
v = df2.Metadata_labels.value_counts()
small_df = df2[df2.Metadata_labels.isin(v.index[v.gt(1)])]
utils.createUmap(small_df, small_df.shape[0], mode='old')

# %% Calculate Percent Replicating
print('Calculating Percent Replicating')

save_name = "MLP_nR3_Atvsplit"  # "TVsplit_allWells_gene_nR3"  ||  "TVsplit_OnlyControls_well_nR3"
group_by_feature = 'Metadata_pert_iname'
n_replicates = 3
n_samples = 10000

corr_replicating_df = pd.DataFrame()
for plates in [[Tplate1, Tplate2], [Vplate1, Vplate2]]:
    temp_df = CalculatePercentReplicating(plates, group_by_feature, n_replicates, n_samples)
    corr_replicating_df = pd.concat([corr_replicating_df, temp_df], ignore_index=True)

print(corr_replicating_df[['Description', 'Percent_Replicating']].to_markdown(index=False))

utils_benchmark.distribution_plot(df=corr_replicating_df, output_file=f"{save_name}_PR.png",
                                  metric="Percent Replicating")

corr_replicating_df['Percent_Replicating'] = corr_replicating_df['Percent_Replicating'].astype(float)

plot_corr_replicating_df = (
    corr_replicating_df.rename(columns={'Modality': 'Perturbation'})
        .drop(columns=['Null_Replicating', 'Value_95', 'Replicating'])
)
