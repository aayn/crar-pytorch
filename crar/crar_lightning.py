import inspect
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
import numpy as np
from box import Box
from crar.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from crar.agent import CRARAgent
from crar.data import ReplayBuffer, Experience, ExperienceDataset
from crar.environments import SimpleMaze
from crar.plotting import plot_maze_abstract_transitions
from crar.utils import nonthrowing_issubclass
from crar.losses import (
    compute_mf_loss,
    compute_trans_loss,
    compute_ld1_loss,
    compute_ld1_prime_loss,
    compute_ld2_loss,
)


class CRARLightning(pl.LightningModule):

    # Store all the PyTorch optimizers in a dictionary with their names as key
    OPTIMIZERS = {
        k: v
        for k, v in inspect.getmembers(torch.optim)
        if nonthrowing_issubclass(v, torch.optim.Optimizer)
    }

    def __init__(self, hparams: Box):
        super().__init__()
        self.hparams = hparams
        if hparams.is_atari:
            self.env = wrap_pytorch(wrap_deepmind(make_atari(hparams.env)))
        elif hparams.is_custom:
            self.env = SimpleMaze(higher_dim_obs=True)
        else:
            self.env = gym.make(hparams.env)

        self.optimizer = CRARLightning.OPTIMIZERS[self.hparams.optimizer.name]

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

        # direction along which to align action 2 transitions
        self.interp_vector = torch.tensor([[-1.0, 0]], device=self.device)

        self.total_reward = 0
        self.episode_reward = 0
        # self.latest_loss = None
        self.populate(hparams.warm_start_size)

    def reset(self):
        self.state = self.env.reset()

    @torch.no_grad()
    def play_step(self):
        eps = 1.0
        # eps = max(
        #     self.hparams.eps_end,
        #     self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame,
        # )
        action = self.agent.act(np.expand_dims(self.state, 0), eps)
        next_state, reward, done, _ = self.env.step(action)
        exp = Experience(self.state, action, reward, done, next_state)
        self.replay_buffer.push(exp)
        self.state = next_state
        if done or (self.global_step + 1) % 500 == 0:
            self.reset()
        return reward, done

    def populate(self, steps: int = 1000) -> None:
        """Fills up the replay buffer by running for `steps` steps."""
        for _ in range(steps):
            self.play_step()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.agent.get_value(x)

    def compute_mf_loss(self, batch):
        return compute_mf_loss(self.agent, batch, self.hparams)

    def compute_trans_loss(self, encoded_batch):
        return compute_trans_loss(self.agent, encoded_batch)

    def compute_ld1_loss(self):
        return compute_ld1_loss(
            self.agent, self.replay_buffer, self.hparams, self.device
        )

    def compute_ld2_loss(self):
        return compute_ld2_loss(
            self.agent, self.replay_buffer, self.hparams, self.device
        )

    def compute_ld1_prime_loss(self, encoded_batch):
        return compute_ld1_prime_loss(encoded_batch)

    def compute_interp_loss(self, encoded_batch):
        encoded_states, *_ = encoded_batch
        actions = torch.tensor([0] * self.hparams.batch_size, device=self.device)
        predicted_next_states = self.agent.compute_transition(encoded_states, actions)
        transition = predicted_next_states - encoded_states

        return -(nn.CosineSimilarity()(transition, self.interp_vector)).sum()

    def training_step(self, batch, batch_number, optimizer_idx=None):
        states, actions, rewards, dones, next_states = batch
        encoded_batch = (
            self.agent.encode(states),
            actions,
            rewards,
            dones,
            self.agent.encode(next_states),
        )

        # DQ optimizer
        if optimizer_idx == 0:
            # print(optimizer_idx)
            if self.global_step % self.hparams.sync_rate == 0:
                self.agent.synchronize_networks()

            if not self.hparams.is_custom:
                reward, done = self.play_step()
                self.episode_reward += reward

                if done:
                    self.total_reward = self.episode_reward
                    self.episode_reward = 0

            else:
                self.episode_reward = 0

            mf_loss = self.compute_mf_loss(batch)
            # loss = mf_loss

            if self.trainer.use_dp or self.trainer.use_ddp2:
                mf_loss = mf_loss.unsqueeze(0)

            tqdm_dict = {"mf_loss": mf_loss}
            output = OrderedDict(
                {"loss": mf_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )

        elif optimizer_idx == 1:
            ld1_loss = self.compute_ld1_loss()
            if self.trainer.use_dp or self.trainer.use_ddp2:
                ld1_loss = ld1_loss.unsqueeze(0)

            tqdm_dict = {"ld1_loss": ld1_loss}
            output = OrderedDict(
                {"loss": ld1_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )

        elif optimizer_idx == 2:
            ld2_loss = self.compute_ld2_loss()
            if self.trainer.use_dp or self.trainer.use_ddp2:
                ld2_loss = ld2_loss.unsqueeze(0)

            tqdm_dict = {"ld2_loss": ld2_loss}
            output = OrderedDict(
                {"loss": ld2_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )

        elif optimizer_idx == 3:
            ld1_prime_loss = self.compute_ld1_prime_loss(encoded_batch)
            if self.trainer.use_dp or self.trainer.use_ddp2:
                ld1_prime_loss = ld1_prime_loss.unsqueeze(0)

            tqdm_dict = {"ld1_prime_loss": ld1_prime_loss}
            output = OrderedDict(
                {"loss": ld1_prime_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )

        elif optimizer_idx == 4:
            transition_loss = self.compute_trans_loss(encoded_batch)
            if self.trainer.use_dp or self.trainer.use_ddp2:
                transition_loss = transition_loss.unsqueeze(0)

            tqdm_dict = {"transition_loss": transition_loss}
            output = OrderedDict(
                {"loss": transition_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
        elif optimizer_idx == 5:
            interp_loss = self.compute_interp_loss(encoded_batch)
            if self.trainer.use_dp or self.trainer.use_ddp2:
                interp_loss = interp_loss.unsqueeze(0)

            tqdm_dict = {"interp_loss": interp_loss}
            output = OrderedDict(
                {"loss": interp_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )

        return output

    def on_epoch_end(self):
        if self.hparams.is_custom:
            all_inputs = self.env.all_possible_inputs()
            encoded_inputs = self.agent.encode(all_inputs)
            plot_maze_abstract_transitions(
                all_inputs,
                encoded_inputs.detach().cpu().numpy(),
                self,
                self.global_step,
                self.hparams.plot_dir,
            )

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize optimizers."""

        dq_optimizer = self.optimizer(
            list(self.agent.encoder.parameters())
            + list(self.agent.current_qnet.parameters()),
            **self.hparams.optimizer.params,
        )
        representation_optimizer1 = self.optimizer(
            self.agent.encoder.parameters(), **self.hparams.optimizer.params
        )

        representation_optimizer2 = self.optimizer(
            self.agent.encoder.parameters(), **self.hparams.optimizer.params
        )

        representation_optimizer3 = self.optimizer(
            self.agent.encoder.parameters(),
            lr=self.hparams.optimizer.params["lr"] / 5.0,
        )

        transition_optimizer = self.optimizer(
            list(self.agent.encoder.parameters())
            + list(self.agent.transition_predictor.parameters()),
            **self.hparams.optimizer.params,
        )

        interp_optimizer = self.optimizer(
            list(self.agent.encoder.parameters())
            + list(self.agent.transition_predictor.parameters()),
            lr=self.hparams.optimizer.params["lr"] / 2.0,
        )

        dq_sched = optim.lr_scheduler.ExponentialLR(dq_optimizer, 0.97)
        repr_sched1 = optim.lr_scheduler.ExponentialLR(representation_optimizer1, 0.97)
        repr_sched2 = optim.lr_scheduler.ExponentialLR(representation_optimizer2, 0.97)
        repr_sched3 = optim.lr_scheduler.ExponentialLR(representation_optimizer3, 0.97)
        trans_sched = optim.lr_scheduler.ExponentialLR(transition_optimizer, 0.97)
        interp_sched = optim.lr_scheduler.ExponentialLR(interp_optimizer, 0.97)

        return (
            [
                dq_optimizer,
                representation_optimizer1,
                representation_optimizer2,
                representation_optimizer3,
                transition_optimizer,
                interp_optimizer,
            ],
            [
                dq_sched,
                repr_sched1,
                repr_sched2,
                repr_sched3,
                trans_sched,
                interp_sched,
            ],
        )

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceDataset(
            self.replay_buffer,
            self.hparams.episode_length * self.hparams.batch_size,
            replace=True,
        )
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
