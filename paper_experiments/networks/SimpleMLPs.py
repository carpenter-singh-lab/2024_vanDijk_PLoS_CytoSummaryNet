import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSymmetricNet(nn.Module):
    def __init__(self, input_dim=1324, latent_dim=1024, output_dim=512, k=1, dropout=0):
        super(DeepSymmetricNet, self).__init__()
        # make sure that the number of parameters is roughly the same as in the MLPs
        self.dropout = nn.Dropout(dropout)
        self.k = k

        # layers
        self.fc1 = nn.Linear(input_dim, int(256 // k))
        self.fc1s = nn.Linear(int(256 // k), latent_dim)
        self.bn1 = torch.nn.BatchNorm1d(int(256 // k))
        self.bn1s = torch.nn.BatchNorm1d(latent_dim)

        self.proj_layers_seq = nn.Sequential(
            nn.Linear(latent_dim, int(128 // k)),
            nn.LeakyReLU(),
            nn.Linear(int(128 // k), output_dim),
            nn.LeakyReLU(),
        )

        # initializations
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc1s.weight)
        torch.nn.init.xavier_uniform_(self.proj_layers_seq.weight)

    def forward(self, x):
        x1 = self.bn1(self.fc1(x))
        x2 = self.bn1s(self.fc1s(x)).sum(dim=1).repeat(1, 1, int(256 // self.k))
        x = F.relu(x1 + x2)

        features = torch.sum(x, 1, keepdim=True)
        x = self.proj_layers_seq(features)

        return x, features


class MLPmean(nn.Module):
    def __init__(self, input_dim=1938, latent_dim=256, output_dim=128, k=1):
        super(MLPmean, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Feature extraction sub-model
        self.lin1 = nn.Linear(
            input_dim, int(256 // k)
        )  # (input channels, output channels, kernel_size)
        self.lin2 = nn.Linear(int(256 // k), int(256 // k))

        self.lin3 = nn.Linear(
            int(256 // k), self.latent_dim
        )  # this projects the BSx1938 vector into a BSxlatent_dim vector

        # Projection head on top of the desired feature representation
        self.proj1 = nn.Linear(self.latent_dim, int(128 // k))
        self.proj2 = nn.Linear(int(128 // k), int(128 // k))
        self.proj3 = nn.Linear(int(128 // k), self.output_dim)

    def forward(self, x):
        # Feature extraction sub-model
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = torch.mean(x, 1, keepdim=True)
        x = x.view(x.shape[0], -1)
        features = F.leaky_relu(self.lin3(x))

        # Projection head
        x = F.leaky_relu(self.proj1(features))
        x = F.leaky_relu(self.proj2(x))
        x = F.leaky_relu(self.proj3(x))

        return x, features


class MLPsum(nn.Module):
    def __init__(self, input_dim=1324, latent_dim=1024, output_dim=512, k=1, dropout=0):
        super(MLPsum, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Feature extraction sub-model
        self.lin1 = nn.Linear(
            input_dim, int(256 // k)
        )  # (input channels, output channels, kernel_size)
        self.lin2 = nn.Linear(int(256 // k), int(256 // k))
        self.lin3 = nn.Linear(
            int(256 // k), self.latent_dim
        )  # this projects the BSx1938 vector into a BSxlatent_dim vector

        # Projection head on top of the desired feature representation
        self.proj1 = nn.Linear(self.latent_dim, int(128 // k))
        self.proj2 = nn.Linear(int(128 // k), int(128 // k))
        self.proj3 = nn.Linear(int(128 // k), self.output_dim)

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Feature extraction sub-model
        x = self.dropout(x)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = torch.sum(x, 1, keepdim=True)
        x = x.view(x.shape[0], -1)
        features = F.leaky_relu(self.lin3(x))

        # Projection head
        x = F.leaky_relu(self.proj1(features))
        x = F.leaky_relu(self.proj2(x))
        x = F.leaky_relu(self.proj3(x))

        return x, features


class MLPsumV2(nn.Module):
    def __init__(
        self,
        input_dim=1324,
        latent_dim=1024,
        output_dim=512,
        k=1,
        dropout=0,
        cell_layers=2,
        proj_layers=3,
        reduction="sum",
    ):
        super(MLPsumV2, self).__init__()

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)
        self.reduction = reduction

        # Pre cell collapse sub-model
        if cell_layers == 1:
            self.cell_layers_seq = nn.Sequential(
                nn.Linear(input_dim, latent_dim), nn.LeakyReLU()
            )
        elif cell_layers == 2:
            self.cell_layers_seq = nn.Sequential(
                nn.Linear(input_dim, int(256 // k)),
                nn.LeakyReLU(),
                nn.Linear(int(256 // k), latent_dim),
                nn.LeakyReLU(),
            )
        elif cell_layers == 3:
            self.cell_layers_seq = nn.Sequential(
                nn.Linear(input_dim, int(256 // k)),
                nn.LeakyReLU(),
                nn.Linear(int(256 // k), int(256 // k)),
                nn.LeakyReLU(),
                nn.Linear(int(256 // k), latent_dim),
                nn.LeakyReLU(),
            )
        elif cell_layers == 4:
            self.cell_layers_seq = nn.Sequential(
                nn.Linear(input_dim, int(256 // k)),
                nn.LeakyReLU(),
                nn.Linear(int(256 // k), int(256 // k)),
                nn.LeakyReLU(),
                nn.Linear(int(256 // k), int(256 // k)),
                nn.LeakyReLU(),
                nn.Linear(int(256 // k), latent_dim),
                nn.LeakyReLU(),
            )

        # This is where the perm invariant operation takes place

        # Projection head on top of the desired feature representation
        if proj_layers == 1:
            self.proj_layers_seq = nn.Sequential(
                nn.Linear(latent_dim, output_dim), nn.LeakyReLU()
            )
        elif proj_layers == 2:
            self.proj_layers_seq = nn.Sequential(
                nn.Linear(latent_dim, int(128 // k)),
                nn.LeakyReLU(),
                nn.Linear(int(128 // k), output_dim),
                nn.LeakyReLU(),
            )
        elif proj_layers == 3:
            self.proj_layers_seq = nn.Sequential(
                nn.Linear(latent_dim, int(128 // k)),
                nn.LeakyReLU(),
                nn.Linear(int(128 // k), int(128 // k)),
                nn.LeakyReLU(),
                nn.Linear(int(128 // k), output_dim),
                nn.LeakyReLU(),
            )

    def forward(self, x):
        x = self.dropout(x)

        # Feature extraction sub-model
        activations = self.cell_layers_seq(x)

        # Collapse cell dimension
        if self.reduction == "sum":
            x = torch.sum(activations, 1, keepdim=True)
        elif self.reduction == "mean":
            x = torch.mean(activations, 1, keepdim=True)
        features = x.view(x.shape[0], -1)
        # Projection head
        x = self.proj_layers_seq(features)

        return x, activations


if __name__ == "__main__":
    M = MLPsumV2(1324, 2048, 2048, 0.5, 0, 1, 2, "sum")
    print([p.numel() for p in M.parameters() if p.requires_grad])
    total_parameters = sum(p.numel() for p in M.parameters() if p.requires_grad)
    print("Total number of parameters:", total_parameters)
