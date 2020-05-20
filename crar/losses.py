import numpy as np
import torch
import torch.nn as nn


def compute_mf_loss(agent, batch, hparams=None):
    """Computes the model-free double Q-learning loss."""

    states, actions, rewards, dones, next_states = batch
    encoded_states = agent.encode(states)

    state_action_values = (
        agent.get_value(encoded_states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    )

    with torch.no_grad():
        next_state_values = agent.get_value(
            agent.encode(next_states, from_current=False), from_current=False
        ).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * hparams.gamma + rewards

    return nn.MSELoss()(state_action_values, expected_state_action_values)


def compute_trans_loss(agent, encoded_batch, hparams=None):
    """Computes the loss for the transition model."""

    encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch
    predicted_next_states = agent.compute_transition(encoded_states, actions)

    return nn.MSELoss()(predicted_next_states, encoded_next_states)


def compute_ld1_loss(agent, replay_buffer, hparams=None, device="cpu"):
    """Computes the disambiguation loss between random states."""

    random_states_1 = replay_buffer.sample(hparams.batch_size)[0]
    random_states_2 = np.roll(random_states_1, 1, axis=0)

    random_states_1 = agent.encode(torch.as_tensor(random_states_1, device=device))
    random_states_2 = agent.encode(torch.as_tensor(random_states_2, device=device))

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


def compute_ld1_prime_loss(encoded_batch, hparams=None):
    """Computes the disabmiguation loss between consecutive states."""

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


def compute_ld2_loss(agent, replay_buffer, hparams=None, device="cpu"):
    """Computes the loss that to regulates the size of the abstract features."""

    random_states_1 = replay_buffer.sample(hparams.batch_size)[0]

    random_states_1 = agent.encode(torch.as_tensor(random_states_1, device=device))
    ld2 = torch.clamp(torch.max(torch.pow(random_states_1, 2)) - 1.0, 0.0, 100.0)

    return ld2
