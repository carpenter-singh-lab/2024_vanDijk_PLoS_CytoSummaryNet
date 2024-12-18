import pandas as pd
import os
import glob
import numpy as np
src.utils import sqlite_to_df
from sklearn.preprocessing import StandardScaler
import pickle
import gc
import argparse
import string
import time

def preprocessLINCS(args):
    # Intialize paths
    sqlite_paths = glob.glob(os.path.join(args.datadir, '*.sqlite'))
    barcode_plate_map = pd.read_csv(os.path.join(args.metadatadir, args.metadata_filename))

    # Start preprocessing
    print('Start preprocessing')
    for path in sqlite_paths:
        platebarcode = path.split('/')[-1].split('.')[0]
        plate_map_name = \
            barcode_plate_map.loc[barcode_plate_map['Assay_Plate_Barcode'] == platebarcode]['Plate_Map_Name'].iloc[0]
        print('Processing plate', platebarcode, 'using metadata file', plate_map_name)

        df = sqlite_to_df(path, metadata_path=os.path.join(args.metadatadir, 'platemap', plate_map_name + '.txt'),
                          compute_subsample=args.subsample, only_load_high_dosepoints=args.only_load_high_dosepoints)  # Compute_subsample==True will return a chunk of 1000 cells
        st = time.time()
        print('Successfully retrieved dataframe')
        feature_column_names = df.columns[~df.columns.str.contains("Metadata")].tolist()
        metadata_column_names = df.columns[df.columns.str.contains("Metadata")].tolist()

        # Load into pandas dataframe
        features = df[feature_column_names]
        metadata = df[metadata_column_names]

        del df

        # Filter out NaNs
        features = features.dropna().to_numpy(np.float32)
        # %% Normalize per plate
        print('Normalizing plate...')
        gc.collect()

        scaler = StandardScaler(copy=False).fit(features)
        features = scaler.transform(features)

        # Generate row of wells to iterate over
        wells = pd.DataFrame({"well_position": [a + y for a in list(string.ascii_uppercase)[:16] for y in list(map(lambda x: "{:0=2d}".format(x), list(range(1, 25, 1))))]})

        output_dirName = f'datasets/{args.dataset}/DataLoader_{platebarcode}'

        # return to pandas DF for speed
        print('Concatenating...')
        df = pd.concat([metadata, pd.DataFrame(features, columns=feature_column_names)], axis=1)

        del features, metadata

        try:
            os.mkdir('datasets')
        except:
            pass
        try:
            os.mkdir(f'datasets/{args.dataset}')
        except:
            pass
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
                    'cell_features': pd.DataFrame(np.zeros((1, 1781)))
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

        et = time.time()
        print('Preprocessing time model is ', et - st)
    print('Finished preprocessing')

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
    # Optional positional argument
    parser.add_argument('subsample', nargs='?', const=False,
                        help='Compute a subsample of the first 1000 cells in the sqlite file. Usefull for debugging.')
    # Optional positional argument
    parser.add_argument('only_load_high_dosepoints', nargs='?', const=True,
                        help='Only load data with dose pointts larger than 3 uM.')
    # Optional positional argument
    parser.add_argument('sqlite_shell_script_path', nargs='?',
                        help='Path to script containing all shell commands to download sqlite files.')

    # Parse arguments
    args = parser.parse_args()

    print("Argument values:")
    print(args.dataset)
    print(args.datadir)
    print(args.metadatadir)
    print(args.metadata_filename)
    print(args.subsample)
    print(args.sqlite_shell_script_path)

    ##% Run powershell command to download sqlite files
    f = open(args.sqlite_shell_script_path, 'r')
    file = f.readlines()
    f.close()

    # Check which files are already downloaded
    plates = [k.split('.')[0].split('/')[-1] for k in file if len(k) > 20]
    existing_plates = [k.split('_')[-1] for k in glob.glob(os.path.join('datasets', args.dataset, 'DataLoader_*'))]
    non_existing_plates = list(set(plates) - set(existing_plates))

    l = []
    for line in file:
        l.append(any(substring in line for substring in non_existing_plates))
    not_downloaded_files = np.array(file)[np.array(l)]

    commands = [line.strip('\n') for line in not_downloaded_files if len(line) > 20][1:]

    for cmd in commands:
        # Download sqlite file
        os.system(cmd)
        print("Downloaded sqlite file to:", cmd.split(' ')[-1])
        # Preprocess it into pickle files
        preprocessLINCS(args)
        # Remove the sqlite file
        os.remove(cmd.split(' ')[-1])
        print(f"Deleted plate {cmd.split(' ')[-1]}")

    print("Finished preprocessing all files!")




