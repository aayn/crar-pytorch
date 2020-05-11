import abc
import numpy as np
import torch
import torch.nn as nn
import gym
from itertools import accumulate


class Encoder(nn.Module):
    def __init__(self, input_shape, device, act, abstract_state_dim=5):
        super().__init__()
        self.input_shape = input_shape
        self.device = device
        self.convs = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=2),
            act(),
            nn.Conv2d(32, 64, kernel_size=2),
            act(),
            # nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3),
            act(),
            # nn.MaxPool2d(3),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 200),
            act(),
            nn.Linear(200, 100),
            act(),
            nn.Linear(100, 50),
            act(),
            nn.Linear(50, 10),
            act(),
            nn.Linear(10, abstract_state_dim),
        )

    def forward(self, x):
        x = self.convs(torch.as_tensor(x, device=self.device).float())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.convs(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)


class SimpleEncoder(nn.Module):
    """Encoder for the CartPole-v0 environment for rapid testing."""

    def __init__(self, input_shape, device, act, abstract_state_dim=5):
        super().__init__()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, abstract_state_dim),
        )

    def forward(self, x):
        # print(self.device)
        x = self.fc(torch.as_tensor(x, device=self.device).float())
        return x
