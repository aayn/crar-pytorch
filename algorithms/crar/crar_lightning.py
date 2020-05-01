import pytorch_lightning as pl
import argparse
from collections import OrderedDict, deque
import gym
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from wrappers import make_atari, wrap_deepmind, wrap_pytorch
from agent import CRARAgent
from data import ReplayBuffer, Experience, ExperienceDataset
import numpy as np
from box import Box


class CRARLightning(pl.LightningModule):
    def __init__(self, hparams: Box):
        super().__init__()
        self.hparams = hparams
        if hparams.is_atari:
            self.env = wrap_pytorch(wrap_deepmind(make_atari(hparams.env)))
        else:
            self.env = gym.make(hparams.env)

        self.reset()

        self.device = self.get_device()

        self.replay_buffer = ReplayBuffer(self.hparams.replay_size)

        self.agent = CRARAgent(
            self.env,
            self.replay_buffer,
            self.device,
            encoder_act=nn.Tanh,
            qnet_act=nn.Tanh,
            rp_act=nn.Tanh,
            abstract_state_dim=self.hparams.abstract_state_dim,
        )

        self.total_reward = 0
        self.episode_reward = 0
        self.latest_loss = None
        self.populate(hparams.warm_start_size)

    def reset(self):
        self.state = self.env.reset()

    @torch.no_grad()
    def play_step(self):
        eps = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame,
        )
        action = self.agent.act(np.expand_dims(self.state, 0), eps)
        next_state, reward, done, _ = self.env.step(action)
        exp = Experience(self.state, action, reward, done, next_state)
        self.replay_buffer.push(exp)
        self.state = next_state
        if done:
            self.reset()
        return reward, done

    def populate(self, steps: int = 1000) -> None:
        """Fills up the replay buffer by running for `steps` steps."""
        for _ in range(steps):
            self.play_step()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.agent.get_value(x)

    def mf_loss(self, encoded_batch):
        encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch

        state_action_values = (
            self.agent.get_value(encoded_states)
            .gather(1, actions.unsqueeze(-1))
            .squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.agent.get_value(encoded_next_states, False).max(1)[
                0
            ]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def trans_loss(self, encoded_batch):
        encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch
        transition = self.agent.compute_transition(encoded_states, actions)

        return nn.MSELoss()(encoded_states + transition, encoded_next_states)

    def representation_loss(self, encoded_batch):
        # TODO: Recheck representation loss
        ld1 = 0
        ld2 = 0
        Cd = 5
        random_states_1 = self.agent.encoder(
            torch.as_tensor(
                self.replay_buffer.sample(len(encoded_batch))[0], device=self.device
            )
        )
        random_states_2 = self.agent.encoder(
            torch.as_tensor(
                self.replay_buffer.sample(len(encoded_batch))[0], device=self.device
            )
        )

        ld1 = torch.exp(-Cd * torch.norm(random_states_1 - random_states_2)) / len(
            encoded_batch
        )
        # TODO: Try other alternatives if not mean
        ld2 = torch.max(torch.norm(random_states_1, p=float("inf")) - 1, 0)[0].mean()

        encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch
        beta = 0.2
        ld1_ = torch.exp(-Cd * torch.norm(encoded_states - encoded_next_states))

        return ld1 + beta * ld1_ + ld2

    def training_step(self, batch, batch_number, optimizer_idx):
        states, actions, rewards, dones, next_states = batch
        encoded_batch = (
            self.agent.encoder(states),
            actions,
            rewards,
            dones,
            self.agent.encoder(next_states),
        )

        reward, done = self.play_step()
        self.episode_reward += reward

        mf_loss = self.mf_loss(encoded_batch)
        trans_loss = self.trans_loss(encoded_batch)
        representation_loss = self.representation_loss(encoded_batch)

        loss = mf_loss + trans_loss + representation_loss

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        if self.global_step % self.hparams.sync_rate == 0:
            self.agent.synchronize_networks()
        log = {
            "total_reward": torch.tensor(self.total_reward).to(self.device),
            "reward": torch.tensor(reward).to(self.device),
            "mf_loss": mf_loss,
            "trans_loss": trans_loss,
            "train_loss": loss,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(self.device),
            "total_reward": torch.tensor(self.total_reward).to(self.device),
        }

        return OrderedDict(
            {
                "mf_loss": mf_loss,
                "trans_loss": trans_loss,
                "loss": loss,
                "log": log,
                "progress_bar": status,
            }
        )

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        dq_optimizer = optim.RMSprop(
            self.agent.current_qnet.parameters(), lr=self.hparams.qnet_lr
        )
        encoder_optimizer = optim.RMSprop(
            self.agent.encoder.parameters(), lr=self.hparams.qnet_lr
        )
        transition_optimizer = optim.RMSprop(
            self.agent.transition_predictor.parameters(), lr=self.hparams.qnet_lr
        )

        return [dq_optimizer, encoder_optimizer, transition_optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceDataset(self.replay_buffer, self.hparams.episode_length)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size,)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
