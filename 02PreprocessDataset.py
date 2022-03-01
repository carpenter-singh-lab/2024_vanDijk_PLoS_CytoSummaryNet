"""
Import large .csv file(s) (>20 GB) and preprocess them so that they can be used during training/validation/testing
"""

import numpy as np
import pandas as pd
import os
import pickle

plate = 'BR00113818_FS'

dirpath = r'/Users/rdijk/Documents/Data/ProcessedData/Stain2'
file = f'{plate}.parquet'
filename = os.path.join(dirpath, file)

df = pd.read_parquet(filename, engine='pyarrow')  # Specify offending columns' data type

# Some row of wells to iterate over
filename2 = '/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv'
wells = pd.read_csv(filename2, usecols=['well_position'])

output_dirName = f'datasets/Stain2/DataLoader_{plate}'

try:
    os.mkdir(output_dirName)
except:
    pass

for index, row in wells.iterrows():
    print(f"Index: {index}, Value: {row[0]}")
    result = df[df.well_position == row[0]]
    cell_features = result[[c for c in result.columns if c.startswith('Cells') or c.startswith('Nuclei') or c.startswith('Cytoplasm') ]]  # [c for c in result.columns if c.startswith("Cells")]
    well = pd.unique(result.loc[:, 'well_position'])
    pert_iname = pd.unique(result.loc[:, 'pert_iname'])
    pert_type = pd.unique(result.loc[:, 'pert_type'])
    moa = pd.unique(result.loc[:, 'moa'])

    assert well == row[0]
    assert len(pert_iname) == 1
    assert len(pert_type) == 1
    assert len(moa) == 1

    print('Cell features array size: ', np.shape(cell_features))

    dict = {
        'well_position': well[0],
        'pert_iname': pert_iname[0],
        'pert_type': pert_type[0],
        'moa': moa[0],
        'cell_features': cell_features
    }

    with open(os.path.join(output_dirName, f'{well[0]}.pkl'), 'wb') as f:
        pickle.dump(dict, f)

