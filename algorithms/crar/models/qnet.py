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

        # self.convs = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 16, kernel_size=2),
        #     nn.Tanh(),
        #     nn.Conv2d(16, 32, kernel_size=3),
        #     nn.Tanh(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(32, 16, kernel_size=2),
        #     nn.Tanh(),
        #     nn.Conv2d(16, 4, kernel_size=1),
        #     nn.Tanh(),
        # )

        # self.fc = nn.Sequential(
        #     nn.Linear(self.feature_size(), 200),
        #     nn.Tanh(),
        #     nn.Linear(200, 50),
        #     nn.Tanh(),
        #     nn.Linear(50, 20),
        #     nn.Tanh(),
        #     nn.Linear(20, self.num_actions),
        # )

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

    # # def feature_size(self):
    # #     return self.convs(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    # def get_value(self, state):
    #     return self(state)

    # def act(self, state, epsilon):
    #     if np.random.random() > epsilon:
    #         state = torch.tensor(state, device=self.device).unsqueeze(0)
    #         q_value = self(state)
    #         action = torch.argmax(q_value).item()
    #     else:
    #         action = np.random.randint(self.num_actions)
    #     return action


class SimpleQNetwork(nn.Module, QLearner):
    def __init__(self, input_shape, num_actions, device, act):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device

        self.fc = nn.Sequential(
            nn.Linear(input_shape, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, self.num_actions),
        )

    def forward(self, x):
        x = self.fc(x.float())
        return x

    def get_value(self, state):
        return self(state)

    def act(self, state):
        # state = torch.as_tensor(state, device=self.device)
        q_value = self(state)
        action = torch.argmax(q_value).item()
        return action


def synchronize_target_model(
    current_model: torch.nn.Module, target_model: torch.nn.Module
):
    target_model.load_state_dict(current_model.state_dict())
