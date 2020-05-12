import abc
import random
import numpy as np
import torch
import torch.nn as nn
from itertools import accumulate


class RewardPredictor(nn.Module):
    def __init__(self, abstract_state_dim, num_actions, fc):
        super().__init__()
        self.fc = fc

    def forward(self, x):
        x = self.fc(x.float())
        return x
