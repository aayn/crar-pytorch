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
from environments import SimpleMaze


class CRARLightning(pl.LightningModule):
    optimizers = {"Adam": optim.Adam, "RMSprop": optim.RMSprop}

    def __init__(self, hparams: Box):
        super().__init__()
        self.hparams = hparams
        if hparams.is_atari:
            self.env = wrap_pytorch(wrap_deepmind(make_atari(hparams.env)))
        elif hparams.is_custom:
            self.env = SimpleMaze(higher_dim_obs=True)
        else:
            self.env = gym.make(hparams.env)

        self.optimizer = CRARLightning.optimizers[self.hparams.optimizer.name]

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
        # self.latest_loss = None
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

    def compute_mf_loss(self, encoded_batch):
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

    def compute_trans_loss(self, encoded_batch):
        encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch
        transition = self.agent.compute_transition(encoded_states, actions)

        interp_loss = 0.0
        # if self.hparams.is_custom:
        #     x2 = np.array([1, 0])
        #     print(transition.shape)
        #     x2 = np.repeat(x2, transition.shape[0], axis=0)
        #     x2 = torch.as_tensor(x2, device=self.device)
        #     print(x2.shape)
        #     interp_loss = -nn.CosineSimilarity()(transition[0], x2)

        print(f"trans = {transition[0]}")
        print(encoded_states[0])
        print(encoded_next_states[0])
        return (
            nn.MSELoss()(encoded_states + transition, encoded_next_states) + interp_loss
        )

    def compute_representation_loss(self, encoded_batch):
        # TODO: Recheck representation loss
        ld1 = 0
        ld2 = 0
        Cd = 5
        # TODO: Currently doesn't deal with batches
        encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch

        random_states_1 = self.agent.encode(
            torch.as_tensor(
                self.replay_buffer.sample(encoded_states.shape[0])[0],
                device=self.device,
            )
        )

        random_states_2 = self.agent.encode(
            torch.as_tensor(
                self.replay_buffer.sample(encoded_states.shape[0])[0],
                device=self.device,
            )
        )

        # ld1 = torch.exp(-Cd * torch.norm(random_states_1[0] - random_states_2[0]))
        ld1 = torch.exp(
            -Cd
            * torch.sqrt(
                torch.clamp(
                    torch.sum(torch.pow(random_states_1[0] - random_states_2[0], 2)),
                    1e-6,
                    10,
                )
            )
        )
        # TODO: Try other alternatives if not mean
        # ld2 = torch.max(
        #     torch.as_tensor([(torch.norm(random_states_1, p=float("inf")) ** 2) - 1, 0])
        # )
        ld2 = torch.clamp(torch.max(torch.pow(random_states_1, 2)) - 1.0, 0.0, 100.0)
        # print(ld2)
        # ld2 = 1 / (torch.norm(random_states_1) + 1e-3)

        encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch
        beta = 0.2
        ld1_ = torch.exp(
            -Cd
            * torch.sqrt(
                torch.clamp(
                    torch.sum(torch.pow(encoded_states - encoded_next_states, 2)),
                    1e-6,
                    10,
                )
            )
        )

        return ld1 + beta * ld1_ + ld2

    def training_step(self, batch, batch_number, optimizer_idx=None):
        # if optimizer_idx in (0,):
        # print(optimizer_idx)
        states, actions, rewards, dones, next_states = batch
        encoded_batch = (
            self.agent.encode(states),
            actions,
            rewards,
            dones,
            self.agent.encode(next_states),
        )

        if self.global_step % self.hparams.sync_rate == 0:
            self.agent.synchronize_networks()

        reward, done = self.play_step()
        self.episode_reward += reward

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        mf_loss = self.compute_mf_loss(encoded_batch)
        trans_loss = self.compute_trans_loss(encoded_batch)
        representation_loss = self.compute_representation_loss(encoded_batch)

        print(f"mf loss {mf_loss}")
        print(f"trans loss {trans_loss}")
        print(f"repr loss {representation_loss}")

        loss = mf_loss + 10 * trans_loss + 10 * representation_loss

        # self.mf_loss, self.trans_loss, self.representation_loss, self.loss = (
        #     mf_loss,
        #     trans_loss,
        #     representation_loss,
        #     loss,
        # )

        # else:
        #     mf_loss, trans_loss, representation_loss, loss = (
        #         self.mf_loss,
        #         self.trans_loss,
        #         self.representation_loss,
        #         self.loss,
        #     )
        # loss = trans_loss

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        log = {
            "total_reward": torch.tensor(self.total_reward).to(self.device),
            # "reward": torch.tensor(reward).to(self.device),
            "mf_loss": mf_loss,
            "trans_loss": trans_loss,
            "val_loss": loss,
            "loss": loss,
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
                "val_loss": loss,
                "log": log,
                "progress_bar": status,
            }
        )

    # def training_epoch_end(self, outputs):
    #     print("Hello from training_epoch_end")
    #     print(outputs)
    #     return outputs

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        # print(optimizer)
        # for opt in optimizer:
        optimizer.step()
        # for opt in optimizer:
        optimizer.zero_grad()

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Optimizer"""

        # dq_optimizer = self.optimizer(
        #     self.agent.current_qnet.parameters(), **self.hparams.optimizer.params,
        # )
        # encoder_optimizer = self.optimizer(
        #     self.agent.encoder.parameters(), **self.hparams.optimizer.params
        # )
        transition_optimizer = optim.Adam(
            list(self.agent.encoder.parameters())
            + list(self.agent.transition_predictor.parameters())
            + list(self.agent.current_qnet.parameters()),
            lr=self.hparams.lr,
        )

        sched1 = optim.lr_scheduler.ReduceLROnPlateau(transition_optimizer)

        return [transition_optimizer], [sched1]
        # dq_optimizer = optim.RMSprop(
        #     self.agent.current_qnet.parameters(), lr=self.hparams.qnet.lr
        # )
        # encoder_optimizer = optim.RMSprop(
        #     self.agent.encoder.parameters(), lr=self.hparams.qnet_lr
        # )
        # transition_optimizer = optim.RMSprop(
        #     self.agent.transition_predictor.parameters(), lr=self.hparams.qnet_lr
        # )

        # return [(dq_optimizer, encoder_optimizer, transition_optimizer)]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceDataset(
            self.replay_buffer, self.hparams.episode_length * self.hparams.batch_size
        )
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
