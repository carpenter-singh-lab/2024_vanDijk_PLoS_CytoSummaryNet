import pytest
import torch
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock
from cytosummarynet.models import CytoSummaryNet
from cytosummarynet.engine import train_loop
from cytosummarynet.dataset import TemplateDataset, my_collate
from pytorch_metric_learning import losses, distances
from tempfile import TemporaryDirectory
from pathlib import Path


@pytest.fixture
def mock_metadata_df():
    """
    Creates a mock metadata DataFrame with file paths and labels.
    """
    return pd.DataFrame(
        {
            "path": ["mock_file1.pkl", "mock_file2.pkl", "mock_file3.pkl"],
            "Metadata_labels": [0, 0, 1],
        }
    )


@pytest.fixture
def mock_pkl_data():
    """
    Simulates pickle file data.
    """
    return {"cell_features": pd.DataFrame(torch.rand(100, 10).numpy())}


@patch("builtins.open", new_callable=mock_open)
@patch("pickle.load")
def test_integrated_pipeline(
    mock_pickle_load, mock_file, mock_metadata_df, mock_pkl_data
):
    """
    Tests the integration of TemplateDataset, CytoSummaryNet, and train_loop.
    """
    # Use a temporary directory for outputs
    with TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir)

        # Mock pickle loading globally
        mock_pickle_load.return_value = mock_pkl_data

        # Initialize TemplateDataset
        dataset = TemplateDataset(
            metadata_df=mock_metadata_df,
            label_column="Metadata_labels",
            path_column="path",
            nr_cells=50,
            nr_sets=3,
            min_cells_per_well=10,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, collate_fn=my_collate
        )

        # Initialize the model
        model = CytoSummaryNet(
            input_dim=10,
            latent_dim=50,
            output_dim=20,
            scaling_factor=1.0,
            cell_layers=2,
            proj_layers=2,
            reduction="mean",
        )

        # Initialize the loss function
        loss_func = losses.SupConLoss(
            distance=distances.CosineSimilarity(), temperature=0.1
        )

        # Initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Mock wandb for logging
        mock_wandb = MagicMock()
        mock_wandb.log = MagicMock()

        # Save initial model parameters for comparison
        initial_params = [param.clone() for param in model.parameters()]

        # Run the training loop
        device = torch.device("cpu")
        trained_model = train_loop(
            model=model,
            trainloader=dataloader,
            valloader=dataloader,  # Using the same dataloader for simplicity
            loss_func=loss_func,
            optimizer=optimizer,
            device=device,
            epochs=2,
            nr_cells=(500, 100),
            wandb=mock_wandb,
            save_path=str(output_path) + "/",
        )

        # Assertions
        assert trained_model is not None
        assert isinstance(trained_model, torch.nn.Module)

        # Check if model parameters have changed after training
        for initial, trained in zip(initial_params, trained_model.parameters()):
            assert not torch.equal(
                initial, trained
            ), "Model parameters did not update during training!"
