"""A SIREN mapping continuous (x, y) positions to RGB colour."""

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega=1.0, first=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega = omega
        bound = 1 / in_features if first else math.sqrt(6 / in_features) / omega
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, coords):
        return torch.sin(self.omega * self.linear(coords))


class Siren(nn.Module):
    def __init__(self, width=256, omega0=30.0):
        super().__init__()
        self.layers = nn.Sequential(
            SineLayer(2, width, omega=omega0, first=True),
            SineLayer(width, width),
            SineLayer(width, width),
            nn.Linear(width, 3),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            self.layers[-2].weight.uniform_(-math.sqrt(6 / width), math.sqrt(6 / width))
            self.layers[-2].bias.zero_()

    def forward(self, coords):
        return self.layers(coords)


def coordinate_grid(height, width, device="cpu"):
    """Pixel centers, with y pointing up and x adjusted for aspect ratio."""
    x = (2 * (torch.arange(width, device=device) + 0.5) / width - 1) * (width / height)
    y = 1 - 2 * (torch.arange(height, device=device) + 0.5) / height
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)
