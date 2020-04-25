import abc
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from itertools import accumulate
from perf import timeit


class Actor(abc.ABC):
    @abc.abstractmethod
    def get_policy(self, obs):
        pass

    @abc.abstractmethod
    def get_action(self, obs):
        pass

    @abc.abstractmethod
    def get_loss(self, obs, actions, weights):
        pass


class Critic(abc.ABC):
    @abc.abstractmethod
    def get_state_value(self, obs):
        pass

    @abc.abstractmethod
    def get_loss(self, obs, weights):
        pass


class MLP(nn.Module):
    def __init__(
        self, input_dim, num_hidden, hidden_dim, activation, output_dim, device
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.layers = [nn.Linear(input_dim, hidden_dim), activation()]
        for _ in range(num_hidden):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation())
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x.float())


class MLPActor(MLP, Actor):
    def __init__(self, obs_dim, num_hidden, hidden_dim, activation, action_dim, device):
        super().__init__(
            obs_dim, num_hidden, hidden_dim, activation, action_dim, device
        )

    def get_policy(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        logits = self(obs)
        return Categorical(logits=logits)

    def get_action(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        return self.get_policy(obs).sample().item()

    def get_loss(self, obs, actions, weights):
        obs = torch.as_tensor(obs, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        weights = torch.as_tensor(weights, device=self.device)
        logprobs = self.get_policy(obs).log_prob(actions)
        loss = -(logprobs * weights).mean()
        return loss


class MLPCritic(MLP, Critic):
    def __init__(self, obs_dim, num_hidden, hidden_dim, activation, action_dim, device):
        super().__init__(
            obs_dim, num_hidden, hidden_dim, activation, action_dim, device
        )

    def get_state_value(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        logits = self(obs)
        return logits

    def get_loss(self, obs, weights):
        obs = torch.as_tensor(obs, device=self.device)
        weights = torch.as_tensor(weights, device=self.device)
        loss = torch.sqrt(torch.pow(self.get_state_value(obs) - weights, 2).sum())
        return loss


if __name__ == "__main__":
    actor = MLPActor(2, 2, 64, nn.PReLU, 2, "cuda")
    print(actor)
