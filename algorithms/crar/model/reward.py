import abc
import random
import numpy as np
import torch
import torch.nn as nn
from itertools import accumulate


class Reward(nn.Module):
    def __init__(self, abstract_state_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(abstract_state_dim + 1, 10),
            nn.Tanh(),
            nn.Linear(10, 50),
            nn.Tanh(),
            nn.Linear(50, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        x = self.fc(x.float())
        return x
