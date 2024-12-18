import pytest
import torch
from cytosummarynet.models import CytoSummaryNet
import torch.nn as nn


def test_model_initialization():
    model = CytoSummaryNet(
        input_dim=100,
        latent_dim=50,
        output_dim=20,
        scaling_factor=1,
        cell_layers=2,
        proj_layers=3,
        reduction="sum",
    )
    assert model is not None


def test_forward_pass():
    model = CytoSummaryNet(
        input_dim=100,
        latent_dim=50,
        output_dim=20,
        scaling_factor=1,
        cell_layers=2,
        proj_layers=3,
        reduction="mean",
    )
    batch_size, num_cells, input_dim = 32, 10, 100
    x = torch.rand(batch_size, num_cells, input_dim)

    projected, activations = model(x)

    # Check shapes
    assert projected.shape == (batch_size, 20)
    assert activations.shape == (batch_size, num_cells, 50)


def test_invalid_reduction():
    with pytest.raises(AssertionError):
        CytoSummaryNet(input_dim=100, latent_dim=50, output_dim=20, reduction="invalid")


def test_custom_cell_layers():
    model = CytoSummaryNet(
        input_dim=100, latent_dim=50, output_dim=20, cell_layers=[80, 60, 50]
    )
    assert len(model.cell_layers_seq) == 7  # 3 layers + 4 activations


def test_scaling_factor():
    # Test the effect of scaling_factor
    model = CytoSummaryNet(
        input_dim=100, latent_dim=50, output_dim=20, scaling_factor=2.0, cell_layers=2
    )

    # Extract the in_features and out_features of each Linear layer
    layer_sizes = [
        (layer.in_features, layer.out_features)
        for layer in model.cell_layers_seq
        if isinstance(layer, nn.Linear)
    ]

    # Assert intermediate layer sizes are scaled up
    assert layer_sizes[0] == (
        100,
        512,
    )  # First layer: Input size = 100, Output size = 256 * 2 (scaling_factor)

    model = CytoSummaryNet(
        input_dim=100, latent_dim=50, output_dim=20, scaling_factor=0.5, cell_layers=2
    )
    layer_sizes = [
        (layer.in_features, layer.out_features)
        for layer in model.cell_layers_seq
        if isinstance(layer, nn.Linear)
    ]

    # Assert intermediate layer sizes are scaled down
    assert layer_sizes[0] == (
        100,
        128,
    )  # First layer: Input size = 100, Output size = 256 * 0.5 (scaling_factor)
