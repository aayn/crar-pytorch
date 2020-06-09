import abc
import numpy as np
import torch
import torch.nn as nn
import gym
from itertools import accumulate


class Encoder(nn.Module):
    def __init__(
        self, input_shape, device, fc, convs=None, act=nn.Tanh, abstract_state_dim=2
    ):
        super().__init__()
        self.input_shape = input_shape
        self.device = device
        self.convs = convs
        self.fc = fc

    def forward(self, x):
        if self.convs is not None:
            x = self.convs(torch.as_tensor(x, device=self.device).float())
            x = x.view(x.size(0), -1)
        x = self.fc(x.float())
        return x
