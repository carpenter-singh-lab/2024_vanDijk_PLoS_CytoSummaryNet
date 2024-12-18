import time
import torch
from tqdm import tqdm
import numpy as np
import random
import os


def train_loop(
    model,
    trainloader,
    valloader,
    loss_func,
    optimizer,
    device,
    epochs,
    nr_cells,
    wandb,
    save_path,
):
    """Train a PyTorch model with the given parameters."""
    model.to(device)
    best_val = 0

    for e in range(epochs):
        time.sleep(0.5)
        model.train()
        tr_loss = 0.0

        print("Training epoch")
        for idx, (points, labels) in enumerate(tqdm(trainloader)):
            points, labels = points.to(device), labels.to(device)
            print(points.shape)
            print(labels.shape)
            feats, _ = model(points)
            tr_loss_tmp = loss_func(feats, labels)
            tr_loss += tr_loss_tmp.item()

            tr_loss_tmp.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Adjust number of cells if required
            if isinstance(nr_cells, tuple):
                CELLS = int(np.random.normal(nr_cells[0], nr_cells[1], 1)[0])
                while CELLS < 100 or CELLS % 2 != 0:
                    CELLS = int(np.random.normal(nr_cells[0], nr_cells[1], 1)[0])
                trainloader.dataset.nr_cells = CELLS

        tr_loss /= idx + 1
        wandb.log({"Train Loss": tr_loss}, step=e)

        # Validation
        model.eval()
        val_loss, val_mAP = validate_model(model, valloader, loss_func, device)

        print(
            f"Epoch {e}. Train loss: {tr_loss}. Val loss: {val_loss}. Val mAP: {val_mAP}"
        )

        if val_mAP > best_val:
            best_val = val_mAP
            print("Saving best validation model checkpoint...")
            torch.save(model.state_dict(), os.path.join(save_path, "model_bestval.pth"))

        wandb.log(
            {"Val loss": val_loss, "Val mAP": val_mAP, "best_val_mAP": best_val}, step=e
        )

        torch.save(
            {
                "epoch": e,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "tr_loss": tr_loss,
                "val_loss": val_loss,
                "val_mAP": val_mAP,
            },
            os.path.join(save_path, "model_checkpoint.pth"),
        )

    print("Training completed.")
    return model


def validate_model(model, valloader, loss_func, device):
    """Evaluate the model on validation data."""
    model.eval()
    val_loss = 0.0
    MLP_profiles = torch.tensor([], dtype=torch.float32).to(device)
    MLP_labels = torch.tensor([], dtype=torch.int16).to(device)

    with torch.no_grad():
        for points, labels in tqdm(valloader):
            if points.shape[1] == 1:
                continue
            points, labels = points.to(device), labels.to(device)
            feats, _ = model(points)
            MLP_profiles = torch.cat([MLP_profiles, feats])
            MLP_labels = torch.cat([MLP_labels, labels])

        val_loss = loss_func(MLP_profiles, MLP_labels).item()
        val_mAP = calculate_mAP(MLP_profiles, MLP_labels)

    return val_loss, val_mAP


def calculate_mAP(embeddings, labels):
    """Calculate the mean Average Precision (mAP) given embeddings and labels."""
    # Placeholder: Replace with actual mAP calculation logic.
    return random.uniform(0, 1)  # Mock implementation for testing.
