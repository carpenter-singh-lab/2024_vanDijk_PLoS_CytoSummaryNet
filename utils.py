
# info_nce_loss
import torch
import torch.nn.functional as F
import datetime

# percent replicating
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import utils_benchmark
from pycytominer.operations.transform import RobustMAD
from pycytominer import feature_select

# heatmapCorrelation
import seaborn as sns
# Umap
import umap
import umap.plot

# MAP
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import average_precision_score
import copy
from itertools import islice

# Precision at K
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target

# filter noisy data
from dataloader_pickles import DataloaderEvalV5, DataloaderTrainV6
import torch.utils.data as data

import os

def now():
    return str(datetime.datetime.now())+': '

def addDataPathsToMetadata(rootDir, metadata, platePaths):
    if not isinstance(platePaths, list):
        platePaths = [platePaths]
    for i, path in enumerate(platePaths):
        platedir = os.path.join(rootDir, path)
        filenames = [os.path.join(platedir, file) for file in os.listdir(platedir)]
        filenames.sort()
        metadata[f'plate{i+1}'] = filenames

    return metadata



def my_collate_eval(batch):
    data = [item[0] for item in batch]
    data = torch.cat(data, dim=0)
    target = [item[1] for item in batch]
    target = torch.cat(target, dim=0)
    aggregated_data = [item[2] for item in batch]
    aggregated_data = torch.cat(aggregated_data, dim=0)
    agg_target = [item[3] for item in batch]
    agg_target = torch.cat(agg_target, dim=0)
    return [data, target, aggregated_data, agg_target]

def my_collate(batch):
    data = [item[0] for item in batch if item[0] != None]
    data = torch.cat(data, dim=0)
    target = [item[1] for item in batch if item[0] != None]
    target = torch.cat(target, dim=0)
    return [data, target]

def train_val_split(metadata_df, Tsplit=0.8, sort=True):
    df = pd.DataFrame()
    plate_columns = [c for c in metadata_df.columns if c.startswith("plate")]
    plate_columns = [c for c in metadata_df.columns if c.startswith("plate1")]
    if len(plate_columns) < 1:
        raise Warning("Could not find any plate columns in metadata_df.")
    for i, plate in enumerate(plate_columns):
        df = pd.concat([df, metadata_df[['Metadata_labels', plate]] ])
        if i < len(plate_columns)-1:
            df = df.rename(columns={plate: plate_columns[i+1]})

    df = df.rename(columns={plate_columns[i]: 'well_path'})

    split = int(len(df.index)*Tsplit)
    if sort:
        df = df.sort_values(by='Metadata_labels')

    return [df.iloc[:split, :].reset_index(drop=True), df.iloc[split:, :].reset_index(drop=True)]

def filterData(df, filter, encode=None, mode='default'):
    if 'negcon' in filter: # drop all negcon wells
        if mode == 'default':
            df = df[df.control_type != 'negcon']
        elif mode == 'eval':
            df = df[df.Metadata_control_type != 'negcon']
        elif mode == 'LINCS':
            df = df.dropna(subset=['broad_sample'])
        df = df.reset_index(drop=True)
    if encode!=None:
        pd.options.mode.chained_assignment = None  # default='warn'
        obj_df = df[encode].astype('category')
        df['Metadata_labels'] = obj_df.cat.codes

        df = df.reset_index(drop=True)

    return df


def featureSelection(features, bestk = 800):
    features_df = pd.DataFrame(features)
    pcorr = features_df.corr()
    avcorr = np.array(pcorr.mean(axis=1))
    sort_index = np.argsort(abs(avcorr))
    bestindices = sort_index[:bestk]
    return features[:, bestindices]


def filter_noisy_data(plateDirs, rootDir, model, config):
    model.eval()
    TrainLoaders = []
    for plate in plateDirs:
        metadata = pd.read_csv(
            '/Users/rdijk/Documents/Data/RawData/Stain2/JUMP-MOA_compound_platemap_with_metadata.csv', index_col=False)

        platestring = plate.split('_')[-2]
        metadata = addDataPathsToMetadata(rootDir, metadata, plate)

        # Filter the data and create numerical labels
        df_prep = filterData(metadata, 'negcon', encode='pert_iname')
        # Add all data to one DF
        Total, _ = train_val_split(df_prep, 0.8, sort=True)
        valset = DataloaderEvalV5(Total)
        loader = data.DataLoader(valset, batch_size=1, shuffle=False,
                                 drop_last=False, pin_memory=False, num_workers=0)
        MLP_profiles = pd.DataFrame()
        with torch.no_grad():
            for idx, (points, labels) in enumerate(loader):
                feats, _ = model(points)
                c1 = pd.concat([pd.DataFrame(feats), pd.Series(labels)], axis=1)
                MLP_profiles = pd.concat([MLP_profiles, c1])
        MLP_profiles.columns = [f"f{x}" for x in range(MLP_profiles.shape[1] - 1)] + ['Metadata_labels']
        #MLP_profiles['Metadata_pert_iname'] = list(metadata[metadata['control_type'] != 'negcon']['pert_iname'])
        AP = CalculateMAP(MLP_profiles, 'cosine_similarity', groupby='Metadata_labels')

        # GENERATE NEW TRAINLOADERS
        indices = AP[AP.AP < 0.1].index
        print(f'Removing {len(indices)} noisy labels from {plate.split("//")[-1].split("_")[1]}.')
        newTrainTotal = Total.drop(index=indices).reset_index(drop=True)
        gTDF = newTrainTotal.groupby('Metadata_labels')
        trainset = DataloaderTrainV6(newTrainTotal, nr_cells=config['nr_cells'][0], nr_sets=config['nr_sets'], groupDF=gTDF)
        TrainLoaders.append(data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, collate_fn=my_collate,
                                            drop_last=False, pin_memory=False, num_workers=0))

    return TrainLoaders

# From https://github.com/cytomining/pycytominer/pull/228/commits/41a971ec52fe09625e6de8d2fa4e4f3fbeddf620
import os
from cells import SingleCells
from typing import Any, Sequence

def sqlite_to_df(
    data_path: str,
    metadata_path: str = None,
    image_cols: Sequence = ["TableNumber", "ImageNumber", "Image_Metadata_Site", "Image_FileName_CellOutlines"],
    strata: Sequence = ["Image_Metadata_Plate", "Image_Metadata_Well"],
    compute_subsample: bool = False,
    compression_options: Any = None,
    float_format: Any = None,
    single_cell_normalize: bool = False,
    normalize_args: Any = None,
    metadata_identifier: str = "Metadata_",
    metadata_merge_on: Sequence = ["Metadata_Well"],
    only_load_high_dosepoints: bool = True,
):
    """Function to convert SQLite file to Pandas DataFrame.
    only_load_high_dosepoints: only loads dose points >3 """

    # Define test SQL file
    sql_file = os.path.abspath(data_path) #"sqlite:////" +

    if compute_subsample:
        subsample_n = 1000
    else:
        subsample_n = 'all'
    # define dataframe
    ap = SingleCells(
        file_or_conn=sql_file,
        image_cols=image_cols,
        subsample_n=subsample_n,
        strata=strata,
        metadata_path=metadata_path,
    )

    # Merge compartments and meta information into one dataframe
    df_merged_sc = ap.merge_single_cells(
        sc_output_file="none",
        compute_subsample=False,
        compression_options=compression_options,
        float_format=float_format,
        single_cell_normalize=single_cell_normalize,
        normalize_args=normalize_args,
        only_load_high_dosepoints=only_load_high_dosepoints
    )

    # In case metadata is provided, merge into existing dataframe
    if metadata_path:
        if df_merged_sc.columns.str.contains("Metadata_mmoles_per_liter").any():
            print('Metadata already added.')
        else:
            print('Adding metadata to data')
            # Load additional information of file
            df_info = pd.read_csv(metadata_path, sep='\t')
            df_info = df_info.rename(columns={"well_position": "Metadata_Well",
                                              "plate_map_name": "Metadata_plate_map_name",
                                              "broad_sample": "Metadata_broad_sample",
                                              "mmoles_per_liter": "Metadata_mmoles_per_liter"})

            # Select only metadata
            _info_meta = [m for m in df_info.columns if m.startswith(
                metadata_identifier)]

            # Merge single cell dataframe with additional information
            df_merged_sc = df_merged_sc.merge(
                right=df_info[_info_meta], how="left", on=metadata_merge_on
            )

    return df_merged_sc


#######################
#%% EVALUATION STUFF ##
#######################
def createUmap(df, nSamples, mode='default'):
    plt.figure(figsize=(14, 10), dpi=300)
    labels = df.Metadata_labels.iloc[:nSamples]
    if mode == 'default':
        features = df.iloc[:nSamples, :-1]
    elif mode == 'old':
        features = df.iloc[:nSamples, 12:]
    reducer = umap.UMAP()
    embedding = reducer.fit(features)
    umap.plot.points(embedding, labels=labels, theme='fire')
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP')
    plt.show()
    return


def compoundCorrelation(df, Ncompounds=20):
    plt.figure(figsize=(14, 10), dpi=300)
    df = df.transpose()
    df = df.iloc[:, :Ncompounds]
    Var_Corr = df.corr()
    #plot the heatmap
    sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
    plt.title('compound Correlation')
    plt.show()
    return


def featureCorrelation(df, Nfeatures=20):
    plt.figure(figsize=(14, 10), dpi=300)
    df = df.iloc[:, :Nfeatures]
    Var_Corr = df.corr()
    #plot the heatmap
    sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
    plt.title('feature Correlation')
    plt.show()
    return

def CalculatePercentReplicating(dfs, group_by_feature, n_replicates, n_samples=10000,
                                description='Unknown', percent_matching=False):
    """

    :param dfs: list of plate dataframes that are analysed together.
    :param group_by_feature: feature column which is used to make replicates
    :param n_replicates: number of expected replicates present in the given dataframes 'dfs' based on the 'group_by_feature' column
    :param n_samples: number of samples used to calculate the null distribution, often 10000
    :return: dataframe consisting of the calculated metrics
    """
    # Set plotting and sampling parameters
    random.seed(9000)
    plt.style.use("seaborn-ticks")
    plt.rcParams["image.cmap"] = "Set1"
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

    corr_replicating_df = pd.DataFrame()

    # Merge dfs in list
    try:
        data_df = pd.concat(dfs)
    except:
        data_df = dfs
    print('Created df of size: ', data_df.shape)
    metadata_df = utils_benchmark.get_metadata(data_df)
    features_df = utils_benchmark.get_featuredata(data_df).replace(np.inf, np.nan).dropna(axis=1, how="any")

    data_df = pd.concat([metadata_df, features_df], axis=1)

    replicating_corr = list(utils_benchmark.corr_between_replicates(data_df, group_by_feature, percent_matching))  # signal distribution
    null_replicating = list(utils_benchmark.corr_between_non_replicates(data_df, n_samples=n_samples, n_replicates=n_replicates,
                                                              metadata_compound_name=group_by_feature))  # null distribution

    prop_95_replicating, value_95_replicating = utils_benchmark.percent_score(null_replicating,
                                                                    replicating_corr,
                                                                    how='right')


    corr_replicating_df = corr_replicating_df.append({'Description': f'{description}',
                                                      'Replicating': replicating_corr,
                                                      'Null_Replicating': null_replicating,
                                                      'Percent_Replicating': '%.1f' % prop_95_replicating,
                                                      'Value_95': value_95_replicating}, ignore_index=True)

    return corr_replicating_df



def CalculateMAP(df, distance='euclidean', groupby='Metadata_labels', percent_matching=False):
    df = df.sort_values(by=groupby)

    if percent_matching:
        df.dropna(subset=[groupby], inplace=True)
        df.reset_index(drop=True, inplace=True)

    features = utils_benchmark.get_featuredata(df)

    if distance == 'cosine_similarity':
        dist = pd.DataFrame(cosine_similarity(features))
    elif distance == 'euclidean':
        dist = pd.DataFrame(euclidean_distances(features))

    compound_names = pd.Series(list(df[groupby]))
    dist.set_axis(compound_names, axis=1, inplace=True)
    dist.set_axis(compound_names, axis=0, inplace=True)

    np.fill_diagonal(dist.values, -1)
    well_APs = []
    PatRs = []

    if percent_matching:
        df.reset_index(drop=True, inplace=True)

    iterator = dist.iterrows()
    for index, row in iterator:
        if percent_matching:
            row.reset_index(drop=True, inplace=True)
            current_compound = df['Metadata_pert_iname'][row == -1].values[0]
            indices1 = set(df['Metadata_pert_iname'][df['Metadata_pert_iname']==current_compound].index)
            indices2 = set(df['Metadata_pert_iname'][df[groupby] == index].index)
            sister_indices = indices2 - indices1
            if len(sister_indices) == 0:
                compound_names = compound_names.drop(list(indices1))
                next(islice(iterator, len(indices1)-2, None), '')
                continue
            row.drop(list(indices1), inplace=True)
            labels = copy.deepcopy(row)
            labels.values[:] = 0
            labels.loc[list(sister_indices)] = 1  # set other compound but with same index to 1 only

            row.reset_index(drop=True, inplace=True)
            labels.reset_index(drop=True, inplace=True)
        else:
            row = row[row != -1]
            labels = copy.deepcopy(row)
            labels.values[:] = 0
            labels[index] = 1

        # Calculate AP
        AP = average_precision_score(labels, row)
        AP_corrected = AP - (sum(labels)/len(labels))  # correct AP by subtracting the random baseline
        well_APs.append(AP_corrected)
        # Calculate P@R
        PatR = precision_at_k(labels, row, k=int(sum(labels)))
        PatRs.append(PatR)

    scores = pd.DataFrame(zip(compound_names, well_APs, PatRs), columns=['compound', 'AP', 'precision at R'])

    # plt.figure(figsize=(14, 10), dpi=300)
    # # plot the heatmap
    # sns.heatmap(dist, xticklabels=compound_names, yticklabels=compound_names, annot=True)
    # plt.title('Cosine Similarity compounds')
    # plt.show()
    return scores


def precision_at_k(y_true, y_score, k, pos_label=1):


    y_true_type = type_of_target(y_true)
    if not (y_true_type == "binary"):
        raise ValueError("y_true must be a binary column.")

    # Makes this compatible with various array types
    y_true_arr = column_or_1d(y_true)
    y_score_arr = column_or_1d(y_score)

    y_true_arr = y_true_arr == pos_label

    desc_sort_order = np.argsort(y_score_arr)[::-1]
    y_true_sorted = y_true_arr[desc_sort_order]
    y_score_sorted = y_score_arr[desc_sort_order]

    true_positives = y_true_sorted[:k].sum()

    return true_positives / k

"""
# From https://github.com/ltrottier/ZCA-Whitening-Python/blob/master/zca.py
MIT License
Copyright (c) 2018 Ludovic Trottier
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import kneed
import scipy
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array


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


class ZCA_corr(BaseEstimator, TransformerMixin):
    def __init__(self, copy=False, regularization=None):
        self.copy = copy
        self.eigenvals = []
        self.regularization = regularization

    def estimate_regularization(self, eigenvalue):
        x = [_ for _ in range(len(eigenvalue))]
        kneedle = kneed.KneeLocator(
            x, eigenvalue, S=1.0, curve='convex', direction='decreasing')
        reg = eigenvalue[kneedle.elbow]/10.0
        return reg  # The complex part of the eigenvalue is ignored

    def fit(self, X, y=None):
        """
        Compute the mean, sphering and desphering matrices.
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to compute the mean, sphering and desphering
            matrices.
        """
        X = check_array(X, accept_sparse=False, copy=self.copy, ensure_2d=True)
        X = as_float_array(X, copy=self.copy)
        self.mean_ = X.mean(axis=0)
        X_ = X - self.mean_
        cov = np.dot(X_.T, X_) / (X_.shape[0] - 1)
        V = np.diag(cov)
        df = pd.DataFrame(X_)
        # replacing nan with 0 and inf with large values
        corr = np.nan_to_num(df.corr())
        G, T, _ = scipy.linalg.svd(corr)
        self.eigenvals = T
        if not self.regularization:
            regularization = self.estimate_regularization(T.real)
            self.regularization = regularization
        else:
            regularization = self.regularization
        t = np.sqrt(T.clip(regularization))
        t_inv = np.diag(1.0 / t)
        v_inv = np.diag(1.0/np.sqrt(V.clip(1e-3)))
        self.sphere_ = np.dot(np.dot(np.dot(G, t_inv), G.T), v_inv)
        return self

    def transform(self, X, y=None, copy=None):
        """
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to sphere along the features axis.
        """
        check_is_fitted(self, "mean_")
        X = as_float_array(X, copy=self.copy)
        return np.dot(X - self.mean_, self.sphere_.T)