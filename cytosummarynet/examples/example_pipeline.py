import torch
import pandas as pd
from cytosummarynet.models import CytoSummaryNet
from cytosummarynet.engine import train_loop
from cytosummarynet.dataset import TemplateDataset, my_collate
from pytorch_metric_learning import losses, distances
from pathlib import Path


def main():
    # 1. Prepare your data
    # Create a DataFrame with paths to your pickle files and their corresponding labels
    metadata_df = pd.DataFrame(
        {
            "path": [
                "data/well1.pkl",
                "data/well2.pkl",
                "data/well3.pkl",
                "data/well4.pkl",
            ],
            "Metadata_labels": [0, 0, 1, 1],  # Binary classification example
        }
    )

    # 2. Create dataset and dataloader
    dataset = TemplateDataset(
        metadata_df=metadata_df,
        label_column="Metadata_labels",
        path_column="path",
        nr_cells=50,  # Number of cells to sample per well
        nr_sets=3,  # Number of cell sets to create per well
        min_cells_per_well=10,  # Minimum number of cells required per well
    )

    # Create train and validation dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=my_collate
    )

    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, collate_fn=my_collate
    )

    # 3. Initialize the model
    model = CytoSummaryNet(
        input_dim=10,  # Dimension of input features per cell
        latent_dim=50,  # Dimension of latent space
        output_dim=20,  # Dimension of final embedding
        scaling_factor=1.0,  # Scaling factor for the model
        cell_layers=2,  # Number of layers in cell encoder
        proj_layers=2,  # Number of layers in projection head
        reduction="mean",  # Reduction method for cell embeddings
    )

    # 4. Setup loss function
    loss_func = losses.SupConLoss(
        distance=distances.CosineSimilarity(), temperature=0.1
    )

    # 5. Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 6. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 7. Setup output directory
    output_dir = Path("output/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 8. Train the model
    trained_model = train_loop(
        model=model,
        trainloader=train_loader,
        valloader=val_loader,
        loss_func=loss_func,
        optimizer=optimizer,
        device=device,
        epochs=50,
        nr_cells=(500, 100),  # (max_cells, min_cells) for training
        wandb=None,  # Set to None if not using Weights & Biases
        save_path=str(output_dir) + "/",
    )

    # 9. Save the trained model
    torch.save(trained_model.state_dict(), output_dir / "final_model.pt")


if __name__ == "__main__":
    main()
