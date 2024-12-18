import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
from typing import List, Tuple


class TemplateDataset(Dataset):
    """
    A PyTorch Dataset for handling imbalanced data with label-based grouping
    and multi-stage sampling for data augmentation.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        label_column: str,
        path_column: str,
        nr_cells: int = 400,
        nr_sets: int = 3,
        min_cells_per_well: int = 10,
    ):
        """
        Initialize the dataset.

        Args:
            metadata_df (pd.DataFrame): DataFrame with metadata and paths to pickle files.
            label_column (str): Column name containing the labels.
            path_column (str): Column name containing paths to pickle files.
            nr_cells (int): Number of cells to sample per well.
            nr_sets (int): Number of samples to draw per label group.
            min_cells_per_well (int): Minimum number of cells required in a well for sampling.
        """
        self.metadata_df = metadata_df
        self.label_column = label_column
        self.path_column = path_column
        self.nr_cells = nr_cells
        self.nr_sets = nr_sets
        self.min_cells_per_well = min_cells_per_well

        # Step 1: Group data by the label column
        # This ensures we can fetch all samples corresponding to a specific label.
        self.groups = self.metadata_df.groupby(self.label_column)

    def __len__(self) -> int:
        """
        Return the number of unique labels/groups in the dataset.
        """
        return len(self.groups)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch a sample by group index.

        Args:
            index (int): Index of the label group to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled features and replicated labels.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        # Step 2: Retrieve all samples from the group corresponding to this label
        group_key = list(self.groups.groups.keys())[index]
        group_samples = self.groups.get_group(group_key)

        # Step 3: Load and filter valid wells for the group
        # This retrieves all file paths for the group, loads their data, and filters out invalid wells.
        samples = self._load_group_samples(group_samples)

        if len(samples) == 0:
            raise ValueError(f"No valid wells found for group {group_key}")

        # Step 4: Apply multi-stage random sampling and augmentation
        # Randomly combine wells and sample cells to create augmented training data.
        sampled_features = self._sample_features(samples)

        # Step 5: Replicate labels for the number of sets created
        labels = torch.tensor([group_key] * self.nr_sets, dtype=torch.int16)

        return sampled_features, labels

    def _load_group_samples(self, group_samples: pd.DataFrame) -> List[dict]:
        """
        Load pickle files for a given group and filter invalid wells.

        Args:
            group_samples (pd.DataFrame): DataFrame containing group metadata.

        Returns:
            List[dict]: List of valid wells with cell features.
        """
        samples = []

        # Load each file in the group
        for _, row in group_samples.iterrows():
            file_path = row[self.path_column]
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            # Step 3.1: Skip wells with insufficient cells
            if data["cell_features"].shape[0] < self.min_cells_per_well:
                continue

            samples.append(data)

        return samples

    def _sample_features(self, samples: List[dict]) -> torch.Tensor:
        """
        Generate augmented samples by randomly combining wells and sampling cells.

        Args:
            samples (List[dict]): List of valid wells for the group.

        Returns:
            torch.Tensor: Augmented feature samples of shape (nr_sets, nr_cells, feature_dim).
        """
        sampled_features = []

        for _ in range(self.nr_sets):
            # Step 4.1: Randomly select 1 or 2 wells to combine
            nr_wells = np.random.randint(1, 3)
            selected_wells = (
                np.random.choice(len(samples), nr_wells, replace=False)
                if len(samples) >= nr_wells
                else [0]  # Fall back to the first well if insufficient wells
            )
            selected_samples = [samples[i] for i in selected_wells]

            # Step 4.2: Sample cells from the selected wells
            temp_features = []
            for sample in selected_samples:
                features = sample["cell_features"]
                features = features.dropna()  # Remove NaNs
                n_cells_to_sample = self.nr_cells // len(selected_samples)
                selected_indices = np.random.choice(
                    features.shape[0], n_cells_to_sample
                )
                temp_features.append(features.iloc[selected_indices, :])

            # Step 4.3: Combine features from selected wells
            combined_features = np.concatenate(temp_features)
            sampled_features.append(combined_features)

        # Step 4.4: Convert the sampled features list to a NumPy array first
        sampled_features = np.array(sampled_features, dtype=np.float32)
        return torch.tensor(sampled_features, dtype=torch.float32)


def my_collate(batch):
    data = [item[0] for item in batch if item[0] is not None]
    data = torch.cat(data, dim=0)
    target = [item[1] for item in batch if item[0] is not None]
    target = torch.cat(target, dim=0)
    return [data, target]
