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
    def act(self, obs):
        ...

    @abc.abstractmethod
    def get_value(self, obs, from_current: bool):
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
        abstract_state_dim: int = 3,
        double_learning: bool = True,
    ):
        super().__init__()
        self.device = device
        self.num_actions = env.action_space.n
        self.double_learning = double_learning
        self.image_space = len(env.observation_space.shape) > 1

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
            # SimpleQNetwork(abstract_state_dim, env.action_space.n, device, qnet_act)
            synchronize_target_model(self.current_qnet, self.target_qnet)
            synchronize_target_model(self.encoder, self.target_encoder)

            self.target_qnet.to(self.device)
            self.target_encoder.to(self.device)

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
        # print(f"Three {obs.shape}")
        # if self.prev_obs is not None:
        #     try:
        #         print(f"Obs diff = {torch.allclose(self.prev_obs, obs)}")
        #     except RuntimeError:
        #         pass
        self.prev_obs = obs
        obs = torch.as_tensor(obs, device=self.device)
        if from_current:
            return self.encoder(obs)
        return self.target_encoder(obs)

    def get_value(self, encoded_state, from_current=True):
        # encoded_state = self.encode(obs)
        if from_current:
            return self.current_qnet.get_value(encoded_state)
        return self.target_qnet.get_value(encoded_state)

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

    # @torch.no_grad()
    def act(self, obs, eps):
        # print(f"Two {obs.shape}")
        if np.random.random() < eps:
            return np.random.randint(self.num_actions)
        return self.current_qnet.act(self.encode(obs))

    def synchronize_networks(self):
        if self.double_learning:
            synchronize_target_model(self.current_qnet, self.target_qnet)
            synchronize_target_model(self.encoder, self.target_encoder)
