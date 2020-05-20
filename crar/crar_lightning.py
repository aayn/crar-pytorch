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
from plotting import plot_maze_abstract_transitions


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
        states, actions, rewards, dones, next_states = batch
        encoded_states = self.agent.encode(states)

        state_action_values = (
            self.agent.get_value(encoded_states)
            .gather(1, actions.unsqueeze(-1))
            .squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.agent.get_value(
                self.agent.encode(next_states, from_current=False), from_current=False
            ).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def compute_trans_loss(self, encoded_batch):
        encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch
        predicted_next_states = self.agent.compute_transition(encoded_states, actions)

        # if self.hparams.is_custom:
        #     x2 = np.array([1, 0])
        #     print(transition.shape)
        #     x2 = np.repeat(x2, transition.shape[0], axis=0)
        #     x2 = torch.as_tensor(x2, device=self.device)
        #     print(x2.shape)
        #     interp_loss = -nn.CosineSimilarity()(transition[0], x2)

        # print(f"trans = {transition[0]}")
        # print(encoded_states[0])
        # print(encoded_next_states[0])
        return nn.MSELoss()(predicted_next_states, encoded_next_states)

    def compute_ld1_loss(self):
        random_states_1 = self.replay_buffer.sample(self.hparams.batch_size)[0]
        random_states_2 = np.roll(random_states_1, 1, axis=0)

        random_states_1 = self.agent.encode(
            torch.as_tensor(random_states_1, device=self.device)
        )
        random_states_2 = self.agent.encode(
            torch.as_tensor(random_states_2, device=self.device)
        )

        ld1 = torch.exp(
            -5
            * torch.sqrt(
                torch.clamp(
                    torch.sum(
                        torch.pow(random_states_1 - random_states_2, 2),
                        dim=1,
                        keepdim=True,
                    ),
                    1e-6,
                    10,
                )
            )
        ).sum()

        return ld1

    def compute_ld2_loss(self):
        random_states_1 = self.replay_buffer.sample(self.hparams.batch_size)[0]

        random_states_1 = self.agent.encode(
            torch.as_tensor(random_states_1, device=self.device)
        )
        ld2 = torch.clamp(torch.max(torch.pow(random_states_1, 2)) - 1.0, 0.0, 100.0)

        return ld2

    def compute_ld1_prime_loss(self, encoded_batch):
        encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch

        beta = 0.2
        ld1_ = torch.exp(
            -5
            * torch.sqrt(
                torch.clamp(
                    torch.sum(
                        torch.pow(encoded_states - encoded_next_states, 2),
                        dim=1,
                        keepdim=True,
                    ),
                    1e-6,
                    10,
                )
            )
        ).sum()

        return beta * ld1_

    def compute_interp_loss(self, encoded_batch):
        encoded_states, *_ = encoded_batch
        actions = torch.tensor([0] * self.hparams.batch_size, device=self.device)
        predicted_next_states = self.agent.compute_transition(encoded_states, actions)
        transition = predicted_next_states - encoded_states

        return -(nn.CosineSimilarity()(transition, self.interp_vector)).sum()

    def compute_representation_loss(self, encoded_batch):
        # TODO: Recheck representation loss
        ld1 = 0
        ld2 = 0
        Cd = 5
        # TODO: Currently doesn't deal with batches

        # # random_states_1, random_states_2 = [[1]], [[1]]
        # # while random_states_1[0][0] == random_states_2[0][0]:
        # #     random_states_1 = self.replay_buffer.sample(encoded_states.shape[0])[0]
        # #     # random_states_2 = self.replay_buffer.sample(encoded_states.shape[0])[0]
        # #     random_states_2 =

        # # print(random_states_1[0])
        # # print(random_states_1[1])

        # # TODO: Try other alternatives if not mean
        # # ld2 = torch.max(
        # #     torch.as_tensor([(torch.norm(random_states_1, p=float("inf")) ** 2) - 1, 0])
        # # )
        # ld2 = torch.clamp(torch.max(torch.pow(random_states_1, 2)) - 1.0, 0.0, 100.0)
        # # print(ld2)
        # # ld2 = 1 / (torch.norm(random_states_1) + 1e-3)

        # encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch

        # return ld1 + beta * ld1_ + ld2

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

        # log = {
        #     "total_reward": torch.tensor(self.total_reward).to(self.device),
        #     # "reward": torch.tensor(reward).to(self.device),
        #     "mf_loss": mf_loss,
        #     "trans_loss": trans_loss,
        #     "val_loss": loss,
        #     "loss": loss,
        # }
        # status = {
        #     "steps": torch.tensor(self.global_step).to(self.device),
        #     "total_reward": torch.tensor(self.total_reward).to(self.device),
        # }

        # ret = OrderedDict(
        #     # {
        #     #     "mf_loss": mf_loss,
        #     #     "trans_loss": trans_loss,
        #     # "loss": 0.0,
        #     #     "val_loss": loss,
        #     #     "log": log,
        #     #     "progress_bar": status,
        #     # }
        # )

        # trans_loss = self.compute_trans_loss(encoded_batch)
        # representation_loss = self.compute_representation_loss(encoded_batch)

        # print(f"mf loss {mf_loss}")
        # print(f"trans loss {trans_loss}")
        # print(f"repr loss {representation_loss}")

        # loss = mf_loss + 10 * trans_loss + 10 * representation_loss

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

    # def training_epoch_end(self, outputs):
    #     print("Hello from training_epoch_end")
    #     print(outputs)
    #     return outputs

    def on_epoch_end(self):
        if self.hparams.is_custom:
            all_inputs = self.env.all_possible_inputs()
            encoded_inputs = self.agent.encode(all_inputs)
            plot_maze_abstract_transitions(
                all_inputs,
                encoded_inputs.detach().cpu().numpy(),
                self,
                self.global_step,
            )

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

        # sched1 = optim.lr_scheduler.ReduceLROnPlateau(transition_optimizer)
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
            [dq_sched, repr_sched1, repr_sched2, repr_sched3, trans_sched, interp_sched]
            # [dq_sched, repr_sched1, repr_sched2, repr_sched3, trans_sched],
        )
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
            self.replay_buffer,
            self.hparams.episode_length * self.hparams.batch_size,
            replace=True,
        )
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
