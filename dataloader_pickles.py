# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:16:34 2022

@author: Robert van Dijk
"""

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

import utils


class DataloaderTrainV4(Dataset):
    """ Dataloader used for loading pickle files on the fly from the laoder during the training.
     Data augmentation is possible, not implemented yet. """

    def __init__(self, df, nr_cells=400):
        """
        Args:
            df: dataframe of all metadata and paths to the pickle files per plate
            nr_cells: number of cells that are sampled from the single-cell feature wells
        """

        self.df = df
        self.nr_cells = nr_cells

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.df.iloc[:, 1][idx], 'rb') as f:
            sample1 = pickle.load(f)
        # extract numpy array
        features = sample1['cell_features']

        label = self.df['Metadata_labels'][idx]

        # Remove possible NaNs
        features = features[~np.isnan(features).any(axis=1)]
        assert ~np.isnan(features).any()

        current_nr_cells = features.shape[0]
        if current_nr_cells > self.nr_cells:
            flag = 1
            if current_nr_cells > 2*self.nr_cells:
                # Randomly select N cells from each array with replicates
                random_indices1 = np.random.choice(current_nr_cells, self.nr_cells)
                sampled_features = features[random_indices1, :]
                # do it again
                random_indices2 = np.random.choice(current_nr_cells, self.nr_cells)
                sampled_features = [sampled_features, features[random_indices2, :]]
                # and again
                random_indices3 = np.random.choice(current_nr_cells, self.nr_cells)
                sampled_features.append(features[random_indices3, :])
                label = torch.tensor([label, label, label], dtype=torch.int16)
            else:
                random_indices1 = np.random.choice(current_nr_cells, self.nr_cells)
                sampled_features = features[random_indices1, :]
                random_indices2 = np.random.choice(current_nr_cells, self.nr_cells)
                sampled_features = [sampled_features, features[random_indices2, :]]
                label = torch.tensor([label, label], dtype=torch.int16)
        else:
            # Randomly select N cells from each array with replicates
            flag = 0
            random_indices = np.random.choice(current_nr_cells, self.nr_cells)
            sampled_features = features[random_indices, :]

        # Normalize per feature
        if flag:
            for i in range(len(sampled_features)):
                scaler = StandardScaler().fit(sampled_features[i])
                sampled_features[i] = scaler.transform(sampled_features[i])
            sampled_features = torch.Tensor(sampled_features)
        else:
            scaler = StandardScaler().fit(sampled_features)
            sampled_features = scaler.transform(sampled_features)
            sampled_features = torch.unsqueeze(torch.Tensor(sampled_features), dim=0)
            label = torch.unsqueeze(torch.tensor(label, dtype=torch.int16), dim=0)

        return [sampled_features, label]


class DataloaderEvalV5(Dataset):
    """ Dataloader used for loading single pickle files for cell feature aggregation (bs=1). """
    def __init__(self, df, feature_selection=False):
        """
        Args:
            df: dataframe of all metadata and paths to the pickle files per plate
        """
        self.df = df
        self.feature_selection = feature_selection

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.df.iloc[:, 1][idx], 'rb') as f:
            sample1 = pickle.load(f)
        # extract numpy array
        features = sample1['cell_features']

        label = self.df['Metadata_labels'][idx]

        # Remove possible NaNs
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        features = features[~np.isnan(features).any(axis=1)]
        assert ~np.isnan(features).any()

        if self.feature_selection:
            features = utils.featureSelection(features, 800)

        scaler = StandardScaler().fit(features)
        features = scaler.transform(features)

        features = torch.tensor(features, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.int16)

        return [features, label]


#%% TODO Development
class DataloaderTrainV5(Dataset):
    """ Dataloader used for loading pickle files on the fly from the laoder during the training.
     Data augmentation is possible, not implemented yet. """

    def __init__(self, df, nr_cells=400, nr_sets=3, feature_selection=False):
        """
        Args:
            df: dataframe of all metadata and paths to the pickle files per plate
            nr_cells: number of cells that are sampled from the single-cell feature wells
            nr_sets: number of cell features sets that are drawn from each well
        """

        self.df = df
        self.nr_cells = nr_cells
        self.nr_sets = nr_sets
        self.feature_selection = feature_selection

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.df.iloc[:, 1][idx], 'rb') as f:
            sample1 = pickle.load(f)
        # extract numpy array
        features = sample1['cell_features']
        label = self.df['Metadata_labels'][idx]
        # Remove possible NaNs
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        features = features[~np.isnan(features).any(axis=1)]
        assert ~np.isnan(features).any()

        if self.feature_selection:
            features = utils.featureSelection(features, 800)

        sampled_features = []
        for _ in range(self.nr_sets):
            random_indices = np.random.choice(features.shape[0], self.nr_cells)
            features_select = features[random_indices, :]
            scaler = StandardScaler().fit(features_select)
            features_select = scaler.transform(features_select)
            sampled_features.append(features_select)
        sampled_features = torch.tensor(sampled_features, dtype=torch.float32)
        labels = torch.tensor([label]*self.nr_sets, dtype=torch.int16)

        return [sampled_features, labels]



#%% TODO Deprecated


class DataloaderTrainV3(Dataset):
    """ Dataloader used for loading pickle files on the fly from the laoder during the training.
     Data augmentation is possible, not implemented yet. """

    def __init__(self, df, nr_cells=300):
        """
        Args:
            df: dataframe of all metadata and paths to the pickle files per plate
            nr_cells: number of cells that are sampled from the single-cell feature wells
        """

        self.df = df
        self.nr_cells = nr_cells

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.df.iloc[:, self.df.shape[1]-3][idx], 'rb') as f:
            sample1 = pickle.load(f)
        with open(self.df.iloc[:, self.df.shape[1]-2][idx], 'rb') as f:
            sample2 = pickle.load(f)
        # extract numpy array
        features1 = sample1['cell_features']
        features2 = sample2['cell_features']


        label = self.df['Metadata_labels'][idx]

        # Remove possible NaNs
        features1 = features1[~np.isnan(features1).any(axis=1)]
        features2 = features2[~np.isnan(features2).any(axis=1)]
        assert ~np.isnan(features1).any()
        assert ~np.isnan(features2).any()

        if features1.shape[0] > self.nr_cells:
            features1 = features1[:self.nr_cells, :]
        else:
            # Randomly select N cells from each array with replicates
            random_indices = np.random.choice(features1.shape[0], self.nr_cells)
            features1 = features1[random_indices, :]

        if features2.shape[0] > self.nr_cells:
            features2 = features2[:self.nr_cells, :]
        else:
            # Randomly select N cells from each array again, note that these are different
            random_indices = np.random.choice(features2.shape[0], self.nr_cells)
            features2 = features2[random_indices, :]


        # Normalize per feature
        scaler = StandardScaler().fit(features1)
        features1 = scaler.transform(features1)
        scaler = StandardScaler().fit(features2)
        features2 = scaler.transform(features2)


        return [torch.tensor(features1), torch.tensor(features2)], [torch.tensor(label, dtype=torch.int16), torch.tensor(label, dtype=torch.int16)]

class DataloaderEvalV4(Dataset):
    """ Dataloader used for loading pickle files on the fly from the laoder during the training.
     Data augmentation is possible, not implemented yet. """

    def __init__(self, df, nr_cells=400, multi_sample=True):
        """
        Args:
            df: dataframe of all metadata and paths to the pickle files per plate
            nr_cells: number of cells that are sampled from the single-cell feature wells
        """

        self.df = df
        self.nr_cells = nr_cells
        self.multi_sample = multi_sample

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.df.iloc[:, 1][idx], 'rb') as f:
            sample1 = pickle.load(f)
        # extract numpy array
        features = sample1['cell_features']

        label = self.df['Metadata_labels'][idx]
        agg_label = torch.unsqueeze(torch.tensor(label, dtype=torch.int16), dim=0)

        # Remove possible NaNs
        features = features[~np.isnan(features).any(axis=1)]
        assert ~np.isnan(features).any()
        collapsed = np.mean(features, axis=0)

        current_nr_cells = features.shape[0]
        # SAMPLE MULTIPLE POINT SETS FROM THE SAME WELL IF MULTI_SAMPLE
        if current_nr_cells > self.nr_cells and self.multi_sample:
            flag = 1
            if current_nr_cells > 2 * self.nr_cells:
                # Randomly select N cells from each array with replicates
                random_indices1 = np.random.choice(current_nr_cells, self.nr_cells)
                sampled_features = features[random_indices1, :]
                # do it again
                random_indices2 = np.random.choice(current_nr_cells, self.nr_cells)
                sampled_features = [sampled_features, features[random_indices2, :]]
                # and again
                random_indices3 = np.random.choice(current_nr_cells, self.nr_cells)
                sampled_features.append(features[random_indices3, :])
                label = torch.tensor([label, label, label], dtype=torch.int16)

                #aggregated_features = torch.tensor([collapsed, collapsed, collapsed])
            else:
                random_indices1 = np.random.choice(current_nr_cells, self.nr_cells)
                sampled_features = features[random_indices1, :]
                random_indices2 = np.random.choice(current_nr_cells, self.nr_cells)
                sampled_features = [sampled_features, features[random_indices2, :]]
                label = torch.tensor([label, label], dtype=torch.int16)

                #aggregated_features = torch.tensor([collapsed, collapsed])
        else:
            # Randomly select N cells from each array with replicates
            flag = 0
            random_indices = np.random.choice(current_nr_cells, self.nr_cells)
            sampled_features = features[random_indices, :]


        aggregated_features = torch.unsqueeze(torch.tensor(collapsed), dim=0)


        # Normalize per feature
        if flag:
            for i in range(len(sampled_features)):
                scaler = StandardScaler().fit(sampled_features[i])
                sampled_features[i] = scaler.transform(sampled_features[i])
            sampled_features = torch.tensor(sampled_features, dtype=torch.float)
        else:
            scaler = StandardScaler().fit(sampled_features)
            sampled_features = scaler.transform(sampled_features)
            sampled_features = torch.unsqueeze(torch.tensor(sampled_features, dtype=torch.float), dim=0)
            label = torch.unsqueeze(torch.tensor(label, dtype=torch.int16), dim=0)

        return [sampled_features, label, aggregated_features, agg_label]