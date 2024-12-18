import torch
import torch.nn as nn
from typing import List, Union


class CytoSummaryNet(nn.Module):
    """
    Modular and parameterizable CytoSummaryNet architecture for feature extraction and projection with pooling.
    """

    def __init__(
        self,
        input_dim: int = 1324,
        latent_dim: int = 1024,
        output_dim: int = 512,
        scaling_factor: float = 1.0,
        dropout: float = 0.0,
        cell_layers: Union[int, List[int]] = 2,
        proj_layers: Union[int, List[int]] = 3,
        reduction: str = "sum",
        activation: nn.Module = nn.LeakyReLU,
    ):
        """
        Args:
            input_dim (int): Input feature size.
            latent_dim (int): Output size of the cell feature extractor.
            output_dim (int): Output size of the projection head.
            scaling_factor (float): Factor to scale intermediate layer sizes.
            dropout (float): Dropout rate.
            cell_layers (int or List[int]): Number of layers or list of layer sizes for the cell extractor.
            proj_layers (int or List[int]): Number of layers or list of layer sizes for the projection head.
            reduction (str): Pooling method ('sum', 'mean', or 'max').
            activation (nn.Module): Activation function.
        """
        super(CytoSummaryNet, self).__init__()

        # Validate reduction method
        assert reduction in [
            "sum",
            "mean",
            "max",
        ], "Invalid reduction method. Choose from 'sum', 'mean', or 'max'."
        self.reduction = reduction

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Cell feature extraction layers
        self.cell_layers_seq = self._build_layers(
            input_dim=input_dim,
            output_dim=latent_dim,
            num_layers=cell_layers,
            scaling_factor=scaling_factor,
            activation=activation,
        )

        # Projection head layers
        self.proj_layers_seq = self._build_layers(
            input_dim=latent_dim,
            output_dim=output_dim,
            num_layers=proj_layers,
            scaling_factor=scaling_factor,
            activation=activation,
        )

    def _build_layers(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: Union[int, List[int]],
        scaling_factor: float,
        activation: nn.Module,
    ) -> nn.Sequential:
        """
        Builds a sequential module with the specified number of layers or layer dimensions.

        Args:
            input_dim (int): Input feature size.
            output_dim (int): Output feature size.
            num_layers (int or List[int]): Number of layers or list of layer sizes.
            scaling_factor (float): Scaling factor for intermediate layers.
            activation (nn.Module): Activation function.

        Returns:
            nn.Sequential: Sequential module with the specified layers.
        """
        layers = []

        def _compute_layer_size(base_size: int, scale: float) -> int:
            """Helper function to compute scaled layer sizes."""
            return max(1, int(base_size * scale))  # Ensure layer size is at least 1

        if isinstance(num_layers, int):
            # Generate intermediate layer sizes based on scaling_factor and num_layers
            intermediate_dim = _compute_layer_size(256, scaling_factor)
            for i in range(num_layers - 1):
                layers.append(
                    nn.Linear(
                        input_dim if i == 0 else intermediate_dim, intermediate_dim
                    )
                )
                layers.append(activation())
            layers.append(nn.Linear(intermediate_dim, output_dim))
            layers.append(activation())
        elif isinstance(num_layers, list):
            # Use custom list of layer sizes
            layer_dims = [input_dim] + num_layers + [output_dim]
            for i in range(len(layer_dims) - 1):
                layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                if i < len(layer_dims) - 2:  # Add activation for all but the last layer
                    layers.append(activation())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_cells, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Projected features and cell-wise activations.
        """
        x = self.dropout(x)

        # Feature extraction
        activations = self.cell_layers_seq(x)

        # Pooling operation
        if self.reduction == "sum":
            x = torch.sum(activations, dim=1, keepdim=True)
        elif self.reduction == "mean":
            x = torch.mean(activations, dim=1, keepdim=True)
        elif self.reduction == "max":
            x, _ = torch.max(activations, dim=1, keepdim=True)

        features = x.view(x.shape[0], -1)

        # Projection head
        projected = self.proj_layers_seq(features)

        return projected, activations
