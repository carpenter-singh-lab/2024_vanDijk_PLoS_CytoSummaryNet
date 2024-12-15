# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:16:34 2022

@author: Robert van Dijk
"""

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from pycytominer.operations.transform import RobustMAD
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
from scipy import stats


def clean_cells(df):
    top_corr_feats = ['Cells_AreaShape_MeanRadius',
                      'Cells_AreaShape_MaximumRadius', 'Cells_AreaShape_MedianRadius', 'Cells_AreaShape_Area']
                        # Cytoplasm_Correlation_K_DNA_Brightfield
    bot_corr_feats = ['Cells_Intensity_MeanIntensityEdge_DNA', 'Cytoplasm_Intensity_MeanIntensityEdge_DNA',
                      'Cytoplasm_Intensity_UpperQuartileIntensity_DNA', 'Cytoplasm_Intensity_MeanIntensity_DNA',
                      ]
                        # Cytoplasm_Correlation_K_Brightfield_DNA

    # Remove cells with lowest values (zscore<X) and remove cells with highest values (zscore<X)
    df = df[(stats.zscore(df[top_corr_feats]) > -2).all(axis=1) & (stats.zscore(df[bot_corr_feats]) < 2).all(axis=1)]

    return df


class DataloaderEvalV5(Dataset):
    """ Dataloader used for loading single pickle files for cell feature aggregation (bs=1). """
    def __init__(self, df, preprocess=False, remove_columns=None, remove_noisy_cells=False):
        """
        Args:
            df: dataframe of all metadata and paths to the pickle files per plate
        """
        self.df = df
        self.dataprep = preprocess
        self.remove_columns = remove_columns
        self.remove_noisy_cells = remove_noisy_cells

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.df.iloc[:, 1][idx], 'rb') as f:
            sample1 = pickle.load(f)
        # extract numpy array
        if sample1['cell_features'].shape[1] == 1781 and self.remove_columns is not None:
            if sample1['cell_features'].shape[0] == 1:
                sample1['cell_features'] = np.zeros((1, 1745))
                if self.remove_noisy_cells:
                    sample1['denoised_cell_features'] = np.zeros((1, 1745))
            else:
                sample1['cell_features'] = sample1['cell_features'].drop(self.remove_columns, axis=1)

        elif sample1['cell_features'].shape[1] == 1781 and sample1['cell_features'].shape[0] == 1:
            sample1['cell_features'] = np.zeros((1, 1781))

        if self.remove_noisy_cells and sample1['cell_features'].shape[0] != 1:
            sample1['denoised_cell_features'] = clean_cells(sample1['cell_features'])

        features = sample1['cell_features']
        label = self.df['Metadata_labels'][idx]

        # Remove possible NaNs
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        features = features[~np.isnan(features).any(axis=1)]


        if self.dataprep == 'normalize':
            features -= features.mean()
        elif self.dataprep == 'standardize':
            scaler = StandardScaler(copy=False).fit(features)
            features = scaler.transform(features)
        elif self.dataprep == 'sphere':
            ZCA_fit = ZCA(regularization=0.01).fit(features)
            features = ZCA_fit.transform(features)


        # Append to list of sampled features
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int16)

        if self.remove_noisy_cells:
            if isinstance(sample1['denoised_cell_features'], pd.DataFrame):
                sample1['denoised_cell_features'] = sample1['denoised_cell_features'].to_numpy()
            denoised_features = sample1['denoised_cell_features'][~np.isnan(sample1['denoised_cell_features']).any(axis=1)]
            denoised_features = torch.tensor(denoised_features, dtype=torch.float32)
            return [features, label, denoised_features]

        return [features, label]



class DataloaderTrainV7(Dataset):
    """ First samples from each well and then aggregates, providing an even distribution of cells from each well
    Dataloader used for loading pickle files on the fly from the laoder during the training.
     Data augmentation is possible, not implemented yet. """

    def __init__(self, df, nr_cells=400, nr_sets=3, groupDF=None, compensator=0, remove_columns=None):
        """
        Args:
            df: dataframe of all metadata and paths to the pickle files per plate
            nr_cells: number of cells that are sampled from the single-cell feature wells
            nr_sets: number of cell features sets that are drawn from each well
        """

        self.df = df
        self.nr_cells = nr_cells
        self.groupDF = groupDF
        self.compensator = compensator
        self.nr_sets = nr_sets
        self.remove_columns = remove_columns

    def __len__(self):
        if self.groupDF is not None:
            return len(self.groupDF)
        else:
            return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = idx+self.compensator
        # Load Sample
        if self.groupDF is not None:
            sample = []
            try:
                paths = self.df.iloc[:, 1][self.groupDF.groups[idx]]
            except:
                idx = random.choice(list(self.groupDF.groups.keys()))
                paths = self.df.iloc[:, 1][self.groupDF.groups[idx]]
            for P in paths:
                with open(P, 'rb') as f:
                    s1 = pickle.load(f)
                    if s1['cell_features'].shape[0] == 1:
                        continue
                    if s1['cell_features'].shape[1] == 1781 and self.remove_columns is not None:
                        s1['cell_features'] = s1['cell_features'].drop(self.remove_columns, axis=1)
                    sample.append(s1)
            # Extract numpy array
            label = self.df['Metadata_labels'][self.groupDF.groups[idx]].iloc[0]

        else:
            raise Warning("No definition for groupDF==None")

        if len(sample) == 0:
            return [None, None]

        sampled_features = []
        for _ in range(self.nr_sets):
            # Generate random number of wells to combine into one sample
            nr_wells = np.random.randint(1, 3, 1)  # either 1 or 2 wells combined
            if nr_wells <= len(sample):
                which_wells = np.random.choice(len(sample), nr_wells, replace=False)  # sampling without replacement
            else:
                which_wells = [0]
            temp = [sample[x] for x in which_wells]
            for x in range(len(temp)):
                F = temp[x]['cell_features']
                if F.shape[0] < 10:
                    continue
                F.dropna(inplace=True)  # Remove possible NaNs
                idxs = np.random.choice(F.shape[0], self.nr_cells//len(temp))
                temp[x]['cell_features'] = F.iloc[idxs, :]
            features_select = np.concatenate([z['cell_features'] for z in temp])

            # Append to list of sampled features
            sampled_features.append(features_select)

        sampled_features = np.array(sampled_features) # first convert to numpy array for speed
        sampled_features = torch.tensor(sampled_features, dtype=torch.float32)
        labels = torch.tensor([label]*self.nr_sets, dtype=torch.int16)

        return [sampled_features, labels]


# TODO NOTE THAT THIS IS ONLY USED FOR COVARIANCE EXPERIMENTS -- IT SUBTRACTS THE MEAN PER FEATURE PER WELL
class DataloaderTrainVX(Dataset):
    """ Dataloader used for loading pickle files on the fly from the laoder during the training.
     Data augmentation is possible, not implemented yet. """

    def __init__(self, df, nr_cells=400, nr_sets=3, groupDF=None, compensator=0, preprocess='normalize'):
        """
        Args:
            df: dataframe of all metadata and paths to the pickle files per plate
            nr_cells: number of cells that are sampled from the single-cell feature wells
            nr_sets: number of cell features sets that are drawn from each well
        """

        self.df = df
        self.nr_cells = nr_cells
        self.groupDF = groupDF
        self.compensator = compensator
        self.nr_sets = nr_sets
        self.dataprep = preprocess

    def __len__(self):
        if self.groupDF is not None:
            return len(self.groupDF)
        else:
            return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = idx+self.compensator
        # Load Sample
        if self.groupDF is not None:
            sample = []
            try:
                paths = self.df.iloc[:, 1][self.groupDF.groups[idx]]
            except:
                idx = random.choice(list(self.groupDF.groups.keys()))
                paths = self.df.iloc[:, 1][self.groupDF.groups[idx]]
            for P in paths:
                with open(P, 'rb') as f:
                    s1 = pickle.load(f)
                    s1['cell_features'] = s1['cell_features'][~np.isnan(s1['cell_features']).any(axis=1)]  # Remove possible NaNs
                    # TODO normalize s1 features
                    if self.dataprep == 'normalize':
                        s1['cell_features'] -= s1['cell_features'].mean()
                    elif self.dataprep == 'standardize':
                        scaler = StandardScaler(copy=False).fit(s1['cell_features'])
                        s1['cell_features'] = scaler.transform(s1['cell_features'])
                    elif self.dataprep == 'sphere':
                        ZCA_fit = ZCA(regularization=0.01).fit(s1['cell_features'])
                        s1['cell_features'] = ZCA_fit.transform(s1['cell_features'])
                    sample.append(s1)
            # Extract numpy array
            label = self.df['Metadata_labels'][self.groupDF.groups[idx]].iloc[0]

        else:
            raise Warning("No definition for groupDF==None")

        sampled_features = []
        for _ in range(self.nr_sets):
            # Generate random number of wells to combine into one sample
            nr_wells = np.random.randint(1, 3, 1) # either 1 or 2 wells combined (note that 2 is not exclusive)
            if nr_wells <= len(sample):
                which_wells = np.random.choice(len(sample), nr_wells, replace=False) # sampling without replacement
            else:
                which_wells = [0]
            temp = [sample[x] for x in which_wells]
            features = np.concatenate([x['cell_features'] for x in temp])
            features = features[~np.isnan(features).any(axis=1)]  # Remove possible NaNs
            assert ~np.isnan(features).any()

            if features.shape[0] < 10: # repeat sampling if no cells in well found
                features = np.zeros((1, features.shape[1]))

            # Sample self.nr_cells cells from the (combined) well
            random_indices = np.random.choice(features.shape[0], self.nr_cells)
            features_select = features[random_indices, :]
            # Append to list of sampled features
            sampled_features.append(features_select)

        sampled_features = torch.tensor(sampled_features, dtype=torch.float32)
        labels = torch.tensor([label]*self.nr_sets, dtype=torch.int16)

        return [sampled_features, labels]


#%%
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import as_float_array


class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, copy=False, regularization=None, retain_variance=0.99):
        self.regularization = regularization
        self.eigenvals = None
        self.retain_variance = retain_variance
        self.copy = copy

    def fit(self, X, y=None):
        X = as_float_array(X, copy=self.copy)
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_
        sigma = np.dot(X.T, X) / (X.shape[0] - 1)
        U, S, V = np.linalg.svd(sigma)
        self.eigenvals = S
        if not self.regularization:
            csum = S / S.sum()
            csum = np.cumsum(csum)
            threshold_loc = (csum < self.retain_variance).sum()
            self.regularization = S[threshold_loc]
        tmp = np.dot(U, np.diag(1 / np.sqrt(S + self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        return self

    def transform(self, X):
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed