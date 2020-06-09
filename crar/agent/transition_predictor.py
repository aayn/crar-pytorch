import abc
import random
import numpy as np
import torch
import torch.nn as nn
from itertools import accumulate


class TransitionPredictor(nn.Module):
    def __init__(self, abstract_state_dim, num_actions, fc):
        super().__init__()
        self.abstract_state_dim = abstract_state_dim
        self.num_actions = num_actions
        self.fc = fc

    def forward(self, x):
        tr = self.fc(x.float())
        return x[:, : self.abstract_state_dim] + tr
