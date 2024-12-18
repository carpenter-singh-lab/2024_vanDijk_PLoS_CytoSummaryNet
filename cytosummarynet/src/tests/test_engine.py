import pytest
from unittest.mock import MagicMock
from cytosummarynet.engine import train_loop
from pytorch_metric_learning import losses, distances
import torch
from tempfile import TemporaryDirectory
from pathlib import Path


@pytest.fixture
def mock_dataloader():
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return torch.rand(10), torch.randint(0, 2, ()).item()

    return torch.utils.data.DataLoader(MockDataset(), batch_size=10)


@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super(MockModel, self).__init__()
            self.fc = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x), None

    return MockModel()


@pytest.fixture
def mock_loss_func():
    return losses.SupConLoss(
        distance=distances.CosineSimilarity(), temperature=0.1
    )  # [0.01 - 0.2]


@pytest.fixture
def mock_optimizer(mock_model):
    return torch.optim.Adam(mock_model.parameters(), lr=0.001)


def test_train_loop(mock_dataloader, mock_model, mock_loss_func, mock_optimizer):
    device = torch.device("cpu")
    mock_wandb = MagicMock()
    mock_wandb.log = MagicMock()

    with TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir)

        trained_model = train_loop(
            mock_model,
            mock_dataloader,
            mock_dataloader,  # Using the same dataloader for simplicity
            mock_loss_func,
            mock_optimizer,
            device,
            epochs=2,
            nr_cells=(500, 100),
            wandb=mock_wandb,
            save_path=str(output_path) + "/",
        )
        assert trained_model is not None
        assert isinstance(trained_model, torch.nn.Module)
