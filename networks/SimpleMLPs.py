
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPadapt(nn.Module):

    def __init__(self, pool_size=800, latent_dim=256, output_dim=128, k=1):
        super(MLPadapt, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Feature extraction sub-model
        self.adapt = nn.AdaptiveMaxPool1d(pool_size)
        self.lin1 = nn.Linear(pool_size, int(256//k))  # (input channels, output channels, kernel_size)
        self.lin2 = nn.Linear(int(256//k), int(256//k))

        self.lin3 = nn.Linear(int(256//k), self.latent_dim) # this projects the BSx1938 vector into a BSxlatent_dim vector

        # Projection head on top of the desired feature representation
        self.proj1 = nn.Linear(self.latent_dim, int(128//k))
        self.proj2 = nn.Linear(int(128//k), int(128//k))
        self.proj3 = nn.Linear(int(128//k), self.output_dim)

    def forward(self, x):
        # Feature extraction sub-model
        x = self.adapt(x)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(x.shape[0], -1)
        features = F.leaky_relu(self.lin3(x))

        # Projection head
        x = F.leaky_relu(self.proj1(features))
        x = F.leaky_relu(self.proj2(x))
        x = F.leaky_relu(self.proj3(x))

        return x, features

class MLP(nn.Module):

    def __init__(self, input_dim=1938, latent_dim=256, output_dim=128, k=1):
        super(MLP, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Feature extraction sub-model
        self.lin1 = nn.Linear(input_dim, int(256//k))  # (input channels, output channels, kernel_size)
        self.lin2 = nn.Linear(int(256//k), int(256//k))

        self.lin3 = nn.Linear(int(256//k), self.latent_dim) # this projects the BSx1938 vector into a BSxlatent_dim vector

        # Projection head on top of the desired feature representation
        self.proj1 = nn.Linear(self.latent_dim, int(128//k))
        self.proj2 = nn.Linear(int(128//k), int(128//k))
        self.proj3 = nn.Linear(int(128//k), self.output_dim)

    def forward(self, x):
        # Feature extraction sub-model
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(x.shape[0], -1)
        features = F.leaky_relu(self.lin3(x))

        # Projection head
        x = F.leaky_relu(self.proj1(features))
        x = F.leaky_relu(self.proj2(x))
        x = F.leaky_relu(self.proj3(x))

        return x, features

