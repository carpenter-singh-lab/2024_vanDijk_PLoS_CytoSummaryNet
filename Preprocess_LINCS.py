import pandas as pd
import os
import glob
import numpy as np
from utils import sqlite_to_df
from sklearn.preprocessing import StandardScaler
import pickle
import gc
import argparse

def preprocessLINCS(args):
    # Intialize paths
    sqlite_paths = glob.glob(os.path.join(args.datadir, '*.sqlite'))
    barcode_plate_map = pd.read_csv(os.path.join(args.metadatadir, args.metadata_filename))

    # Start preprocessing
    print('Start')
    for path in sqlite_paths:
        platebarcode = path.split('/')[-1].split('.')[0]
        plate_map_name = \
            barcode_plate_map.loc[barcode_plate_map['Assay_Plate_Barcode'] == platebarcode]['Plate_Map_Name'].iloc[0]
        print('Processing plate', platebarcode, 'using metadata file', plate_map_name)

        df = sqlite_to_df(path, metadata_path=os.path.join(args.metadatadir, 'platemap', plate_map_name + '.txt'),
                          compute_subsample=args.subsample)  # Compute_subsample==True will return a chunk of 1000 cells

        feature_column_names = df.columns[~df.columns.str.contains("Metadata")].tolist()
        metadata_column_names = df.columns[df.columns.str.contains("Metadata")].tolist()

        # Load into pandas dataframe
        features = df[feature_column_names]
        metadata = df[metadata_column_names]

        del df

        # Filter out NaNs
        features = features.dropna().to_numpy()
        # %% Normalize per plate
        print('Normalizing plate...')
        gc.collect()

        scaler = StandardScaler(copy=False).fit(features)
        features = scaler.transform(features)

        # Some row of wells to iterate over
        filename2 = '/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv'
        wells = pd.read_csv(filename2, usecols=['well_position'])

        output_dirName = f'datasets/{args.dataset}/DataLoader_{platebarcode}'

        # return to pandas DF for speed
        print('Concatenating...')
        df = pd.concat([metadata, pd.DataFrame(features, columns=feature_column_names)], axis=1)

        del features, metadata

        try:
            os.mkdir(output_dirName)
        except:
            pass

        for index, row in wells.iterrows():
            print(f"Index: {index}, Value: {row[0]}")
            result = df[df.Metadata_Well == row[0]].copy()
            cell_features = result[feature_column_names]  # [c for c in result.columns if c.startswith("Cells")]
            well = pd.unique(result.loc[:, 'Metadata_Well'])
            broad_sample = pd.unique(result.loc[:, 'Metadata_broad_sample'])
            mmoles_per_liter = pd.unique(result.loc[:, 'Metadata_mmoles_per_liter'])
            plate_map_name = pd.unique(result.loc[:, 'Metadata_plate_map_name'])

            try:  # TODO Check code from here and below
                assert well == row[0]
                assert len(broad_sample) == 1
                assert len(mmoles_per_liter) == 1
                assert len(plate_map_name) == 1
            except:
                print(f'data not available, saving empty well {well}')
                dict = {
                    'well_position': row[0],
                    'cell_features': np.zeros((1, 1))
                }
                with open(os.path.join(output_dirName, f'{row[0]}.pkl'), 'wb') as f:
                    pickle.dump(dict, f)
                continue
            print('Cell features array size: ', np.shape(cell_features))

            dict = {
                'well_position': well[0],
                'broad_sample': broad_sample[0],
                'mmoles_per_liter': mmoles_per_liter,
                'plate_map_name': plate_map_name,
                'cell_features': cell_features
            }

            with open(os.path.join(output_dirName, f'{well[0]}.pkl'), 'wb') as f:
                pickle.dump(dict, f)

    print('Finished')

if __name__=='__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Preprocess LINCS dataset from sqlite file to pickle files containing '
                                                 'single cell data per well.', fromfile_prefix_chars='@')

    # Required positional argument
    parser.add_argument('dataset', type=str,
                        help='Specify the dataset name. It will be used for the output directory path.')
    # Required positional argument
    parser.add_argument('datadir', type=str,
                        help='Specify the directory in which the sqlite files are located.')
    # Required positional argument
    parser.add_argument('metadatadir', type=str,
                        help='Specify the directory in which the metadata plate dirs are located.')
    # Required positional argument
    parser.add_argument('metadata_filename', type=str,
                        help='Specify the barcode platemap filename.')
    # Required positional argument
    parser.add_argument('subsample', nargs='?', const=False,
                        help='Compute a subsample of the first 1000 cells in the sqlite file. Usefull for debugging.')

    # Parse arguments
    args = parser.parse_args()

    print("Argument values:")
    print(args.dataset)
    print(args.datadir)
    print(args.metadatadir)
    print(args.metadata_filename)
    print(args.subsample)

    preprocessLINCS(args)


