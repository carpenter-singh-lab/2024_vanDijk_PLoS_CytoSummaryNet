import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, mock_open
from cytosummarynet.dataset import TemplateDataset


# Mock metadata DataFrame fixture
@pytest.fixture
def mock_metadata_df():
    return pd.DataFrame(
        {
            "path": ["file1.pkl", "file2.pkl", "file3.pkl"],
            "Metadata_labels": [0, 0, 1],
        }
    )


# Mock pickle data fixture
@pytest.fixture
def mock_pkl_data():
    return {
        "cell_features": pd.DataFrame(
            np.random.rand(100, 10)
        )  # 100 cells x 10 features
    }


# Test __init__ method
def test_init(mock_metadata_df):
    dataset = TemplateDataset(
        metadata_df=mock_metadata_df,
        label_column="Metadata_labels",
        path_column="path",
        nr_cells=50,
        nr_sets=3,
    )

    assert dataset.metadata_df.equals(mock_metadata_df)
    assert dataset.label_column == "Metadata_labels"
    assert dataset.path_column == "path"
    assert dataset.nr_cells == 50
    assert dataset.nr_sets == 3
    assert dataset.min_cells_per_well == 10
    assert len(dataset.groups) == 2  # Two unique labels: 0 and 1


# Test __len__ method
def test_len(mock_metadata_df):
    dataset = TemplateDataset(
        metadata_df=mock_metadata_df,
        label_column="Metadata_labels",
        path_column="path",
    )

    assert len(dataset) == len(mock_metadata_df["Metadata_labels"].unique())


# Test _load_group_samples method
@patch("builtins.open", new_callable=mock_open)
@patch("pickle.load")
def test_load_group_samples(
    mock_pickle_load, mock_file, mock_metadata_df, mock_pkl_data
):
    mock_pickle_load.return_value = mock_pkl_data

    dataset = TemplateDataset(
        metadata_df=mock_metadata_df,
        label_column="Metadata_labels",
        path_column="path",
        nr_cells=50,
        nr_sets=3,
    )

    group_samples = dataset.groups.get_group(0)  # Group with label 0
    samples = dataset._load_group_samples(group_samples)

    assert len(samples) == 2  # Two paths for label 0
    assert all("cell_features" in s for s in samples)
    assert all(s["cell_features"].shape == (100, 10) for s in samples)


# Test _sample_features method
@patch("builtins.open", new_callable=mock_open)
@patch("pickle.load")
def test_sample_features(mock_pickle_load, mock_file, mock_metadata_df, mock_pkl_data):
    mock_pickle_load.return_value = mock_pkl_data

    dataset = TemplateDataset(
        metadata_df=mock_metadata_df,
        label_column="Metadata_labels",
        path_column="path",
        nr_cells=50,
        nr_sets=3,
    )

    group_samples = dataset.groups.get_group(0)  # Group with label 0
    samples = dataset._load_group_samples(group_samples)
    sampled_features = dataset._sample_features(samples)

    assert sampled_features.shape == (3, 50, 10)  # 3 sets, 50 cells, 10 features
    assert isinstance(sampled_features, torch.Tensor)


# Test __getitem__ method
@patch("builtins.open", new_callable=mock_open)
@patch("pickle.load")
def test_getitem(mock_pickle_load, mock_file, mock_metadata_df, mock_pkl_data):
    mock_pickle_load.return_value = mock_pkl_data

    dataset = TemplateDataset(
        metadata_df=mock_metadata_df,
        label_column="Metadata_labels",
        path_column="path",
        nr_cells=50,
        nr_sets=3,
    )

    sampled_features, labels = dataset[0]

    assert sampled_features.shape == (3, 50, 10)  # 3 sets, 50 cells, 10 features
    assert labels.shape == (3,)  # 3 replicated labels
    assert labels.tolist() == [0, 0, 0]  # Labels match group 0
