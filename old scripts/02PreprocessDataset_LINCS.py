"""
Import large .csv file(s) (>20 GB) and preprocess them so that they can be used during training/validation/testing
"""

import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd
import glob
import gc
import numpy as np

dataset = 'LINCS'
nr_non_feature_columns = 5

rootDir = r'/Users/rdijk/Documents/ProjectFA/Phase2/Data/ProcessedData'
plateDirs = glob.glob(f'{rootDir}/*.parquet')

for filename in plateDirs:
    plate = filename.split('/')[-1].split('.')[0]
    print(f'Processing {plate}...')
    df = dd.read_parquet(filename, engine='pyarrow')

    features = df.iloc[:, nr_non_feature_columns:]
    feature_column_names = features.columns

    # Load into pandas dataframe
    metadata = df.iloc[:, :nr_non_feature_columns]
    del df

    # Filter out NaNs
    features = features.dropna().compute().to_numpy(dtype='float32')
    #%% Normalize per plate
    print('Normalizing plate...')
    gc.collect()

    scaler = StandardScaler(copy=False).fit(features)
    features = scaler.transform(features)

    # Some row of wells to iterate over
    filename2 = '/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv'
    wells = pd.read_csv(filename2, usecols=['well_position'])

    output_dirName = f'datasets/{dataset}/DataLoader_{plate}'


    # return to pandas DF for speed
    print('Concatenating...')
    df = pd.concat([metadata.compute(), pd.DataFrame(features, columns=feature_column_names)], axis=1)

    del features, metadata

    try:
        os.mkdir(output_dirName)
    except:
        pass


    for index, row in wells.iterrows():
        print(f"Index: {index}, Value: {row[0]}")
        result = df[df.well_position == row[0]].copy()
        cell_features = result[[c for c in result.columns if c.startswith('Cells') or c.startswith('Nuclei') or c.startswith('Cytoplasm') ]]  # [c for c in result.columns if c.startswith("Cells")]
        well = pd.unique(result.loc[:, 'well_position'])
        broad_sample = pd.unique(result.loc[:, 'broad_sample'])
        mg_per_ml = pd.unique(result.loc[:, 'mg_per_ml'])
        mmoles_per_liter = pd.unique(result.loc[:, 'mmoles_per_liter'])
        plate_map_name = pd.unique(result.loc[:, 'plate_map_name'])

        try:
            assert well == row[0]
            assert len(broad_sample) == 1
            assert len(mg_per_ml) == 1
            assert len(mmoles_per_liter) == 1
            assert len(plate_map_name) == 1
        except:
            print(f'data not available, saving empty well {well}')
            dict = {
                'well_position': row[0],
                'cell_features': np.zeros((1,1))
            }
            with open(os.path.join(output_dirName, f'{row[0]}.pkl'), 'wb') as f:
                pickle.dump(dict, f)
            continue
        print('Cell features array size: ', np.shape(cell_features))

        dict = {
            'well_position': well[0],
            'broad_sample': broad_sample[0],
            'mg_per_ml': mg_per_ml[0],
            'mmoles_per_liter': mmoles_per_liter,
            'plate_map_name': plate_map_name,
            'cell_features': cell_features
        }

        with open(os.path.join(output_dirName, f'{well[0]}.pkl'), 'wb') as f:
            pickle.dump(dict, f)

print('Donezo')
