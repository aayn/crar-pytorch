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
    def act(self, state):
        ...


class DuelingQLearner(QLearner):
    @abc.abstractmethod
    def get_state_value(self, state):
        ...

    @abc.abstractmethod
    def get_state_action_value(self, state, action):
        ...

    @abc.abstractmethod
    def get_adv(self, state):
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
        q_value = self(state)
        action = torch.argmax(q_value).item()
        return action


class DuelingQNetwork(nn.Module, DuelingQLearner):
    def __init__(self, input_shape, num_actions, convs, fc, value, advantage, device):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device
        self.convs = convs
        self.fc = fc
        self.value = value
        self.advantage = advantage

    def forward(self, x):
        x = self.fc(x)
        val, adv = self.value(x), self.advantage(x)
        return val, adv

    def get_state_value(self, state):
        val = self.value(self.fc(state))
        return val

    def get_adv(self, state):
        adv = self.advantage(self.fc(state))
        return adv

    def get_value(self, state):
        val, adv = self.forward(state)
        return adv + val

    def get_state_action_value(self, state, action):
        val, adv = self.forward(state)
        state_action_adv = adv.gather(1, action.long().unsqueeze(-1))
        state_action_values = val + (state_action_adv - adv.mean(dim=1).unsqueeze(-1))
        return state_action_values

    def act(self, state):
        adv = self.get_adv(state)
        action = torch.argmax(adv).item()
        return action
