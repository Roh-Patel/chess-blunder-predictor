"""A small residual CNN that combines board planes with scalar features (ratings, ply, clocks)."""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)  # skip connection


class BlunderNet(nn.Module):
    def __init__(self, in_planes: int, n_scalars: int, channels: int = 64, blocks: int = 6,
                 hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.reduce = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.scalar_net = nn.Sequential(nn.Linear(n_scalars, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(32 * 8 * 8 + 64, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, planes, scalars):
        board = self.reduce(self.tower(self.stem(planes))).flatten(1)
        features = torch.cat([board, self.scalar_net(scalars)], dim=1)
        return self.head(features).squeeze(1)  # one logit per position
