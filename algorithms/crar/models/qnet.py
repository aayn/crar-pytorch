import abc
import numpy as np
import torch
import torch.nn as nn
from itertools import accumulate


class QLearner(abc.ABC):
    @abc.abstractmethod
    def get_value(self, state):
        ...

    @abc.abstractmethod
    def act(self, state, epsilon):
        ...


class QNetwork(nn.Module, QLearner):
    def __init__(self, input_shape, num_actions, convs, fc, device):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device
        self.convs = convs
        self.fc = fc

    def forward(self, x):
        if self.convs is not None:
            x = self.convs(x.float())
            x = x.view(x.size(0), -1)
        x = self.fc(x.float())
        return x

    def get_value(self, state):
        return self(state)

    def act(self, state):
        # state = torch.as_tensor(state, device=self.device)
        q_value = self(state)
        action = torch.argmax(q_value).item()
        return action
