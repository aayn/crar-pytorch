import abc
import random
import numpy as np
import torch
import torch.nn as nn
from itertools import accumulate


class Encoder(nn.Module):
    def __init__(self, input_shape, abstract_state_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=2),
            nn.Tanh(),
            nn.Conv2d(8, 16, kernel_size=2),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(3),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 10),
            nn.Tanh(),
            nn.Linear(10, abstract_state_dim),
        )

    def forward(self, x):
        x = self.convs(x.float())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.convs(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)


class SimpleEncoder(nn.Module):
    """Encoder for the CartPole-v0 environment for rapid testing."""

    def __init__(self, input_shape, abstract_state_dim=3):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, abstract_state_dim),
        )

    def forward(self, x):
        x = self.fc(x)
        return x
