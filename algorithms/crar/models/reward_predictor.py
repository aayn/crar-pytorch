import abc
import random
import numpy as np
import torch
import torch.nn as nn
from itertools import accumulate


class RewardPredictor(nn.Module):
    def __init__(self, abstract_state_dim, act):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(abstract_state_dim + 1, 10),
            act(),
            nn.Linear(10, 50),
            act(),
            nn.Linear(50, 20),
            act(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        x = self.fc(x.float())
        return x