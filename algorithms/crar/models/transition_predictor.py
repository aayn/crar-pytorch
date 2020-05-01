import abc
import random
import numpy as np
import torch
import torch.nn as nn
from itertools import accumulate


class TransitionPredictor(nn.Module):
    def __init__(self, abstract_state_dim):
        super().__init__()
        # self.device = device

        self.fc = nn.Sequential(
            nn.Linear(abstract_state_dim + 1, 10),
            nn.Tanh(),
            nn.Linear(10, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 10),
            nn.Tanh(),
            nn.Linear(10, abstract_state_dim),
        )

    def forward(self, x):
        x = self.fc(x.float())
        return x
