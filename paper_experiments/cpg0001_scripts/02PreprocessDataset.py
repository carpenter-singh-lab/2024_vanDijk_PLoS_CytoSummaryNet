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

dataset = "Stain4"
rootDir = rf"/Users/rdijk/Documents/Data/ProcessedData/{dataset}"
plateDirs = glob.glob(f"{rootDir}/*.parquet")

for filename in plateDirs:
    plate = filename.split("/")[-1].split(".")[0]
    print(f"Processing {plate}...")
    df = dd.read_parquet(filename, engine="pyarrow")

    features = df.iloc[:, 6:]
    feature_column_names = features.columns

    # Load into pandas dataframe
    metadata = df.iloc[:, :6]
    del df

    # Filter out NaNs
    features = features.dropna().compute().to_numpy(dtype="float32")
    # %% Normalize per plate
    print("Normalizing plate...")
    gc.collect()

    import matplotlib.pyplot as plt

    plt.scatter(list(range(1324)), features.mean(0), c="red")
    plt.show()

    scaler = StandardScaler(copy=False).fit(features)
    features = scaler.transform(features)

    # Some row of wells to iterate over
    filename2 = "/inputs/cpg0001_metadata/JUMP-MOA_compound_platemap_with_metadata.csv"
    wells = pd.read_csv(filename2, usecols=["well_position"])

    output_dirName = f"datasets/{dataset}/DataLoader_{plate}"

    # return to pandas DF for speed
    print("Concatenating...")
    df = pd.concat(
        [metadata.compute(), pd.DataFrame(features, columns=feature_column_names)],
        axis=1,
    )

    del features, metadata

    try:
        os.mkdir(output_dirName)
    except:
        pass

    for index, row in wells.iterrows():
        print(f"Index: {index}, Value: {row[0]}")
        result = df[df.well_position == row[0]].copy()
        cell_features = result[
            [
                c
                for c in result.columns
                if c.startswith("Cells")
                or c.startswith("Nuclei")
                or c.startswith("Cytoplasm")
            ]
        ]  # [c for c in result.columns if c.startswith("Cells")]
        well = pd.unique(result.loc[:, "well_position"])
        pert_iname = pd.unique(result.loc[:, "pert_iname"])
        pert_type = pd.unique(result.loc[:, "pert_type"])
        moa = pd.unique(result.loc[:, "moa"])

        try:
            assert well == row[0]
            assert len(pert_iname) == 1
            assert len(pert_type) == 1
            assert len(moa) == 1
        except:
            print(f"skipping well {well}")
            continue
        print("Cell features array size: ", np.shape(cell_features))

        dict = {
            "well_position": well[0],
            "pert_iname": pert_iname[0],
            "pert_type": pert_type[0],
            "moa": moa[0],
            "cell_features": cell_features,
        }

        with open(os.path.join(output_dirName, f"{well[0]}.pkl"), "wb") as f:
            pickle.dump(dict, f)

print("Donezo")
