import pickle
import numpy as np
import dask.dataframe as dd
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

import utils

dpath = '/Users/rdijk/PycharmProjects/featureAggregation/datasets/Stain2'
directories = os.listdir(dpath)

hugeDF = []

for D in directories:
    try:
        os.mkdir(os.path.join(dpath, D+'_norm'))
    except:
        pass
    for file in os.listdir(os.path.join(dpath, D)):
        print(file)
        filepath = os.path.join(dpath, D, file)
        with open(os.path.join(filepath), 'rb') as f:
            sample = pickle.load(f)
        features = sample['cell_features']

        if isinstance(hugeDF, list):
            hugeDF = dd.from_pandas(pd.DataFrame(features), chunksize=10000)
        else:
            hugeDF.append(pd.DataFrame(features))

        # scaler = StandardScaler().fit(features)
        # features = scaler.transform(features)
        #
        # sample['cell_features'] = features
        #
        # savepath = os.path.join(dpath, D+'_norm', file)
        # with open(os.path.join(savepath), 'wb') as f:
        #     pickle.dump(sample, f)

    break