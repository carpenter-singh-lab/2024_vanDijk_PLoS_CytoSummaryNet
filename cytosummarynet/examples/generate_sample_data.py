import torch
import pandas as pd
import pickle
from pathlib import Path


def generate_sample_data(
    output_dir: str,
    num_wells: int = 4,
    num_features: int = 10,
    cells_per_well: int = 100,
):
    """
    Generate sample pickle files containing random cell features for testing CytoSummaryNet.

    Args:
        output_dir (str): Directory where pickle files will be saved
        num_wells (int): Number of wells/pickle files to generate
        num_features (int): Number of features per cell
        cells_per_well (int): Number of cells per well

    Returns:
        pd.DataFrame: Metadata DataFrame containing file paths and labels
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize lists for metadata DataFrame
    file_paths = []
    labels = []

    # Generate data for each well
    for i in range(num_wells):
        # Generate random cell features
        cell_features = pd.DataFrame(
            torch.rand(cells_per_well, num_features).numpy(),
            columns=[f"feature_{j}" for j in range(num_features)],
        )

        # Create data dictionary similar to test mock
        data_dict = {"cell_features": cell_features}

        # Create file path
        file_path = output_path / f"well{i+1}.pkl"
        file_paths.append(str(file_path))

        # Assign label (alternating between 0 and 1)
        label = i % 2
        labels.append(label)

        # Save pickle file
        with open(file_path, "wb") as f:
            pickle.dump(data_dict, f)

    # Create metadata DataFrame
    metadata_df = pd.DataFrame({"path": file_paths, "Metadata_labels": labels})

    return metadata_df


# Example usage
if __name__ == "__main__":
    # Generate sample data
    metadata_df = generate_sample_data(
        output_dir="data", num_wells=4, num_features=10, cells_per_well=100
    )

    # Print the metadata
    print("\nGenerated metadata DataFrame:")
    print(metadata_df)

    # Verify the data
    print("\nVerifying first pickle file:")
    with open(metadata_df["path"].iloc[0], "rb") as f:
        data = pickle.load(f)
        print(f"Number of cells: {len(data['cell_features'])}")
        print(f"Number of features: {data['cell_features'].shape[1]}")
