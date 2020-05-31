from crar.models import (
    synchronize_target_model,
    make_encoder,
    make_qnet,
    make_transition_predictor,
    make_reward_predictor,
    make_discount_predictor,
)
import torch
import abc
import numpy as np
import torch.nn as nn
import gym

from crar.data import ReplayBuffer


class AbstractAgent(abc.ABC):
    @abc.abstractmethod
    def encode(self, obs):
        ...

    @abc.abstractmethod
    def act(self, obs, eps):
        ...

    @abc.abstractmethod
    def get_value(self, state, depth: int, from_current: bool):
        ...

    @abc.abstractmethod
    def compute_reward(self, state, actions):
        ...

    @abc.abstractmethod
    def compute_discount(self, state, actions):
        ...


class CRARAgent(nn.Module, AbstractAgent):
    def __init__(
        self,
        env: gym.Env,
        replay_buffer: ReplayBuffer,
        device: torch.device,
        encoder_act=nn.Tanh,
        qnet_act=nn.Tanh,
        rp_act=nn.Tanh,
        discount_factor: float = 0.95,
        abstract_state_dim: int = 3,
        double_learning: bool = True,
        branching_factor: int = 2,
    ):
        super().__init__()
        self.device = device
        self.num_actions = env.action_space.n
        self.discount_factor = discount_factor
        self.double_learning = double_learning
        # TODO: Use if image space to determine if to use Conv or not
        self.image_space = len(env.observation_space.shape) > 1
        self.branching_factor = branching_factor

        self.encoder = make_encoder(
            env.observation_space.shape, abstract_state_dim, device
        )

        # Observation consists of joint values
        # if not self.image_space:
        #     self.encoder = SimpleEncoder(
        #         env.observation_space.shape, device, encoder_act, abstract_state_dim
        #     )
        # # Observation is an image
        # else:
        #     self.encoder = make_encoder(
        #         env.observation_space.shape, abstract_state_dim, device
        #     )
        #     # Encoder(
        #     #     env.observation_space.shape, device, encoder_act, abstract_state_dim
        #     # )

        self.current_qnet = make_qnet(abstract_state_dim, env.action_space.n, device)
        # SimpleQNetwork(abstract_state_dim, env.action_space.n, device, qnet_act)

        self.encoder.to(self.device)
        self.current_qnet.to(self.device)

        if double_learning:
            self.target_qnet = make_qnet(abstract_state_dim, env.action_space.n, device)
            self.target_encoder = make_encoder(
                env.observation_space.shape, abstract_state_dim, device
            )
            self.target_qnet.to(self.device)
            self.target_encoder.to(self.device)
            self.synchronize_networks()

        self.reward_predictor = make_reward_predictor(
            abstract_state_dim, self.num_actions
        )
        # RewardPredictor(abstract_state_dim, rp_act)
        self.reward_predictor.to(self.device)

        self.discount_predictor = make_discount_predictor(
            abstract_state_dim, self.num_actions
        )
        # RewardPredictor(abstract_state_dim, rp_act)
        self.discount_predictor.to(self.device)

        self.transition_predictor = make_transition_predictor(
            abstract_state_dim, self.num_actions
        )
        # TransitionPredictor(
        #     abstract_state_dim, self.num_actions
        # )
        self.transition_predictor.to(device)
        self.prev_obs = None

    def forward(self, x):
        self.get_value(x)

    def encode(self, obs, from_current=True):
        self.prev_obs = obs
        obs = torch.as_tensor(obs, device=self.device)
        if from_current:
            return self.encoder(obs)
        return self.target_encoder(obs)

    def get_value(self, encoded_state, depth=0, from_current=True):
        # encoded_state = self.encode(obs)
        network = self.current_qnet if from_current else self.target_qnet
        if depth == 0:
            return network.get_value(encoded_state)
            # return (
            #     self.current_qnet.get_value(encoded_state)
            #     if from_current
            #     else self.target_qnet.get_value(encoded_state)
            # )

        q_plan_values = []
        for a in range(self.num_actions):
            actions = torch.tensor([a] * encoded_state.shape[0], device=self.device)
            rewards = self.compute_reward(encoded_state, actions)
            # TODO: Decide what to do with discount factor
            discounts = torch.tensor(
                [self.discount_factor] * rewards.shape[0], device=self.device
            )

            # discounts = self.compute_discount(encoded_state, actions)
            next_encoded_states = self.compute_transition(encoded_state, actions)
            q_plan_values.append(
                rewards
                + discounts
                * torch.max(
                    self.get_value(next_encoded_states, depth - 1, from_current), dim=1,
                )[0]
            )
        return torch.cat(q_plan_values, dim=1)

    def compute_transition(self, encoded_state, actions):
        # encoded_state = self.encode(obs)
        # TODO: Decide if use one-hot or not.
        # actions = nn.functional.one_hot(actions, self.num_actions)
        x = torch.cat([encoded_state, actions.float().view(-1, 1)], 1)
        # x = torch.cat([encoded_state, actions.float()], 1)
        return self.transition_predictor(x)

    def compute_reward(self, encoded_state, actions):
        # encoded_state = self.encode(obs)
        # TODO: Decide if use one-hot or not.
        # actions = nn.functional.one_hot(actions, self.num_actions)
        x = torch.cat([encoded_state, actions.float().view(-1, 1)], 1)
        # x = torch.cat([encoded_state, actions.float()], 1)
        return self.reward_predictor(x)

    def compute_discount(self, encoded_state, actions):
        # encoded_state = self.encode(obs)
        # TODO: Decide if use one-hot or not.
        # actions = nn.functional.one_hot(actions, self.num_actions)
        x = torch.cat([encoded_state, actions.float().view(-1, 1)], 1)
        # x = torch.cat([encoded_state, actions.float()], 1)
        return self.discount_predictor(x)

    def act(self, obs, eps, depth=0):
        if np.random.random() < eps:
            return np.random.randint(self.num_actions)
        q_value = self.get_value(self.encode(obs), depth=depth)
        action = torch.argmax(q_value).item()
        return action
        # return self.current_qnet.act(self.encode(obs))

    def synchronize_networks(self):
        if self.double_learning:
            synchronize_target_model(self.current_qnet, self.target_qnet)
            synchronize_target_model(self.encoder, self.target_encoder)
