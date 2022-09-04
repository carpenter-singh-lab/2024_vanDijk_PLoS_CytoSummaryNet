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


class DataloaderEvalV5(Dataset):
    """ Dataloader used for loading single pickle files for cell feature aggregation (bs=1). """
    def __init__(self, df, preprocess=False):
        """
        Args:
            df: dataframe of all metadata and paths to the pickle files per plate
        """
        self.df = df
        self.dataprep = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.df.iloc[:, 1][idx], 'rb') as f:
            sample1 = pickle.load(f)
        # extract numpy array
        features = sample1['cell_features']
        well_position = sample1['well_position']

        label = self.df['Metadata_labels'][idx]

        # Remove possible NaNs
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        features = features[~np.isnan(features).any(axis=1)]
        assert ~np.isnan(features).any()

        if features.shape[0] < 1:
            features = np.zeros((1, features.shape[1]))

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

        return [features, label]

class DataloaderTrainV6(Dataset):
    """ Groups two wells together at random and then samples from then (uneven distribution)
    Dataloader used for loading pickle files on the fly from the laoder during the training.
     Data augmentation is possible, not implemented yet. """

    def __init__(self, df, nr_cells=400, nr_sets=3, groupDF=None, compensator=0):
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
                    sample.append(s1)
            # Extract numpy array
            label = self.df['Metadata_labels'][self.groupDF.groups[idx]].iloc[0]

        else:
            raise Warning("No definition for groupDF==None")

        sampled_features = []
        for _ in range(self.nr_sets):
            # Generate random number of wells to combine into one sample
            nr_wells = np.random.randint(1, 3, 1) # either 1 or 2 wells combined
            if nr_wells <= len(sample):
                which_wells = np.random.choice(len(sample), nr_wells, replace=False) # sampling without replacement
            else:
                which_wells = [0]
            temp = [sample[x] for x in which_wells]
            features = np.concatenate([x['cell_features'] for x in temp])
            features = features[~np.isnan(features).any(axis=1)]  # Remove possible NaNs
            assert ~np.isnan(features).any()

            if features.shape[0] < 10:
                features = np.zeros((1, features.shape[1]))

            # Sample self.nr_cells cells from the (combined) well
            random_indices = np.random.choice(features.shape[0], self.nr_cells)
            features_select = features[random_indices, :]
            # Append to list of sampled features
            sampled_features.append(features_select)

        sampled_features = torch.tensor(sampled_features, dtype=torch.float32)
        labels = torch.tensor([label]*self.nr_sets, dtype=torch.int16)

        return [sampled_features, labels]


class DataloaderTrainV7(Dataset):
    """ First samples from each well and then aggregates, providing an even distribution of cells from each well
    Dataloader used for loading pickle files on the fly from the laoder during the training.
     Data augmentation is possible, not implemented yet. """

    def __init__(self, df, nr_cells=400, nr_sets=3, groupDF=None, compensator=0):
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
                    sample.append(s1)
            # Extract numpy array
            label = self.df['Metadata_labels'][self.groupDF.groups[idx]].iloc[0]

        else:
            raise Warning("No definition for groupDF==None")

        sampled_features = []
        for _ in range(self.nr_sets):
            # Generate random number of wells to combine into one sample
            nr_wells = np.random.randint(1, 3, 1) # either 1 or 2 wells combined
            if nr_wells <= len(sample):
                which_wells = np.random.choice(len(sample), nr_wells, replace=False) # sampling without replacement
            else:
                which_wells = [0]
            temp = [sample[x] for x in which_wells]
            for x in range(len(temp)):
                F = temp[x]['cell_features']
                F.dropna(inplace=True) # Remove possible NaNs
                if F.shape[0] < 10:
                    F = pd.DataFrame(np.zeros((1, F.shape[1])))
                idxs = np.random.choice(F.shape[0], self.nr_cells//len(temp))
                temp[x]['cell_features'] = F.iloc[idxs, :]
            features_select = np.concatenate([x['cell_features'] for x in temp])

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
        scaler = RobustMAD().fit(features1)
        features1 = scaler.transform(features1)
        scaler = RobustMAD().fit(features2)
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
                scaler = RobustMAD().fit(sampled_features[i])
                sampled_features[i] = scaler.transform(sampled_features[i])
            sampled_features = torch.tensor(sampled_features, dtype=torch.float)
        else:
            scaler = RobustMAD().fit(sampled_features)
            sampled_features = scaler.transform(sampled_features)
            sampled_features = torch.unsqueeze(torch.tensor(sampled_features, dtype=torch.float), dim=0)
            label = torch.unsqueeze(torch.tensor(label, dtype=torch.int16), dim=0)

        return [sampled_features, label, aggregated_features, agg_label]
    
    
    
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
                scaler = RobustMAD().fit(sampled_features[i])
                sampled_features[i] = scaler.transform(sampled_features[i])
            sampled_features = torch.Tensor(sampled_features)
        else:
            scaler = RobustMAD().fit(sampled_features)
            sampled_features = scaler.transform(sampled_features)
            sampled_features = torch.unsqueeze(torch.Tensor(sampled_features), dim=0)
            label = torch.unsqueeze(torch.tensor(label, dtype=torch.int16), dim=0)

        return [sampled_features, label]



class DataloaderTrainV5(Dataset):
    """ Dataloader used for loading pickle files on the fly from the laoder during the training.
     Data augmentation is possible, not implemented yet. """

    def __init__(self, df, nr_cells=400, nr_sets=3, groupDF=None, compensator = 0, feature_selection=False):
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
        self.feature_selection = feature_selection

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
            paths = self.df.iloc[:, 1][self.groupDF.groups[idx]]
            for P in paths:
                with open(P, 'rb') as f:
                    s1 = pickle.load(f)
                    sample.append(s1)
            # Extract numpy array
            features = np.concatenate([x['cell_features'] for x in sample])
            label = self.df['Metadata_labels'][self.groupDF.groups[idx]].iloc[0]

        else:
            with open(self.df.iloc[:, 1][idx], 'rb') as f:
                sample1 = pickle.load(f)
                # Extract numpy array
                features = sample1['cell_features']
                label = self.df['Metadata_labels'][idx]

        # Remove possible NaNs
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        features = features[~np.isnan(features).any(axis=1)]
        assert ~np.isnan(features).any()


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