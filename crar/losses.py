import numpy as np
import torch
import torch.nn as nn
from crar.agent import CRARAgent


def compute_disambiguation(tensor1, tensor2, Cd=5.0):
    return torch.exp(
        -Cd * torch.clamp(torch.norm(tensor1 - tensor2, dim=1, keepdim=True), 1e-6, 3.2)
    ).sum()


def compute_ddqn_loss(agent: CRARAgent, batch, hparams=None):
    """Computes the model-free double Q-learning loss."""

    states, actions, rewards, dones, next_states = batch
    encoded_states = agent.encode(states)

    state_action_values = agent.get_value(
        encoded_states, depth=hparams.planning_depth
    ).gather(1, actions.unsqueeze(-1))

    online_next_state_values = agent.get_value(
        agent.encode(next_states), depth=hparams.planning_depth,
    )
    best_online_actions = torch.argmax(online_next_state_values, dim=1)

    target_next_state_values = agent.get_value(
        agent.encode(next_states, from_current=False),
        depth=hparams.planning_depth,
        from_current=False,
    )

    target_state_action_values = target_next_state_values.gather(
        1, best_online_actions.unsqueeze(-1)
    )
    target_state_action_values *= 1.0 - dones.float().unsqueeze(-1)

    expected_state_action_values = (
        rewards.unsqueeze(-1) + hparams.gamma * target_state_action_values
    )

    return nn.MSELoss()(state_action_values, expected_state_action_values)


def compute_dueling_ddqn_loss(agent: CRARAgent, batch, hparams=None):
    states, actions, rewards, dones, next_states = batch
    encoded_states = agent.encode(states)

    state_action_values = agent.current_qnet.get_state_action_value(
        encoded_states, actions
    )

    best_online_actions = torch.argmax(
        agent.current_qnet.get_adv(agent.encode(next_states)), dim=1
    )

    target_state_action_values = agent.target_qnet.get_state_action_value(
        agent.encode(next_states, from_current=False), best_online_actions
    )
    target_state_action_values *= 1.0 - dones.float().unsqueeze(-1)

    expected_state_action_values = (
        rewards.unsqueeze(-1) + hparams.gamma * target_state_action_values
    )

    loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    return loss


def compute_ddqn_loss(agent: CRARAgent, batch, hparams=None):
    """Computes the model-free double Q-learning loss."""

    states, actions, rewards, dones, next_states = batch
    encoded_states = agent.encode(states)

    state_action_values = agent.get_value(
        encoded_states, depth=hparams.planning_depth
    ).gather(1, actions.unsqueeze(-1))

    online_next_state_values = agent.get_value(
        agent.encode(next_states), depth=hparams.planning_depth,
    )
    best_online_actions = torch.argmax(online_next_state_values, dim=1)

    target_next_state_values = agent.get_value(
        agent.encode(next_states, from_current=False),
        depth=hparams.planning_depth,
        from_current=False,
    )

    target_state_action_values = target_next_state_values.gather(
        1, best_online_actions.unsqueeze(-1)
    )
    target_state_action_values *= 1.0 - dones.float().unsqueeze(-1)

    expected_state_action_values = (
        rewards.unsqueeze(-1) + hparams.gamma * target_state_action_values
    )

    return nn.MSELoss()(state_action_values, expected_state_action_values)


def compute_priority_ddqn_loss(agent: CRARAgent, batch, hparams=None):
    priorities, states, actions, rewards, dones, next_states = batch
    encoded_states = agent.encode(states)

    state_action_values = agent.get_value(
        encoded_states, depth=hparams.planning_depth
    ).gather(1, actions.unsqueeze(-1))

    online_next_state_values = agent.get_value(
        agent.encode(next_states), depth=hparams.planning_depth,
    )
    best_online_actions = torch.argmax(online_next_state_values, dim=1)

    target_next_state_values = agent.get_value(
        agent.encode(next_states, from_current=False),
        depth=hparams.planning_depth,
        from_current=False,
    )

    target_state_action_values = target_next_state_values.gather(
        1, best_online_actions.unsqueeze(-1)
    )
    target_state_action_values *= 1.0 - dones.float().unsqueeze(-1)

    expected_state_action_values = (
        rewards.unsqueeze(-1) + hparams.gamma * target_state_action_values
    )

    td_error = expected_state_action_values - state_action_values

    imp_samp_weights = (1 / hparams.replay_size) * (1 / priorities)
    imp_samp_weights /= torch.max(imp_samp_weights)
    imp_samp_weights = torch.pow(imp_samp_weights, hparams.prioritization_beta)
    loss = (td_error).pow(2).mean()

    return td_error, loss


def compute_prioritized_dueling_loss(agent: CRARAgent, batch, hparams=None):
    priorities, states, actions, rewards, dones, next_states = batch
    encoded_states = agent.encode(states)

    state_action_values = agent.current_qnet.get_state_action_value(
        encoded_states, actions
    )

    best_online_actions = torch.argmax(
        agent.current_qnet.get_adv(agent.encode(next_states)), dim=1
    )

    target_state_action_values = agent.target_qnet.get_state_action_value(
        agent.encode(next_states, from_current=False), best_online_actions
    )
    target_state_action_values *= 1.0 - dones.float().unsqueeze(-1)

    expected_state_action_values = (
        rewards.unsqueeze(-1) + hparams.gamma * target_state_action_values
    )

    td_error = expected_state_action_values - state_action_values

    imp_samp_weights = (1 / hparams.replay_size) * (1 / priorities)
    imp_samp_weights /= torch.max(imp_samp_weights)
    imp_samp_weights = torch.pow(imp_samp_weights, hparams.prioritization_beta)
    loss = (td_error).pow(2).mean()

    return td_error, loss


def compute_mf_loss(agent: CRARAgent, batch, hparams=None, mode="ddqn"):
    if mode == "ddqn":
        return compute_ddqn_loss(agent, batch, hparams)
    elif mode == "dueling-ddqn":
        return compute_dueling_ddqn_loss(agent, batch, hparams)
    elif mode == "priority-ddqn":
        td_error, loss = compute_priority_ddqn_loss(agent, batch, hparams)
        agent.replay_buffer.update_priorities(td_error)
        return loss
    elif mode == "priority-dueling":
        td_error, loss = compute_prioritized_dueling_loss(agent, batch, hparams)
        agent.replay_buffer.update_priorities(td_error)
        return loss


def compute_trans_loss(agent, encoded_batch, hparams=None):
    """Computes the loss for the transition model."""

    if hparams.prioritized_buffer:
        _, encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch
    else:
        encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch
    predicted_next_states = agent.compute_transition(encoded_states, actions)

    return nn.MSELoss()(predicted_next_states, encoded_next_states)


def compute_reward_loss(agent, encoded_batch, hparams=None):
    """Computes the loss for the transition model."""

    if hparams.prioritized_buffer:
        _, encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch
    else:
        encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch
    predicted_rewards = agent.compute_reward(encoded_states, actions)
    rewards = rewards.view(-1, 1)
    return nn.MSELoss()(predicted_rewards, rewards)


def compute_ld1_loss(agent, replay_buffer, hparams=None, device="cpu"):
    """Computes the disambiguation loss between random states."""
    if hparams.prioritized_buffer:
        random_states_1 = replay_buffer.sample(hparams.batch_size)[1]
    else:
        random_states_1 = replay_buffer.sample(hparams.batch_size)[0]
    random_states_2 = np.roll(random_states_1, 1, axis=0)

    random_states_1 = agent.encode(torch.as_tensor(random_states_1, device=device))
    random_states_2 = agent.encode(torch.as_tensor(random_states_2, device=device))

    ld1 = compute_disambiguation(random_states_1, random_states_2)
    return ld1


def compute_ld1_prime_loss(encoded_batch, hparams=None):
    """Computes the disabmiguation loss between consecutive states."""
    if hparams.prioritized_buffer:
        _, encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch
    else:
        encoded_states, actions, rewards, dones, encoded_next_states = encoded_batch

    beta = 0.05
    ld1_ = compute_disambiguation(encoded_states, encoded_next_states)
    return beta * ld1_


def compute_ld2_loss(agent, replay_buffer, hparams=None, device="cpu"):
    """Computes the loss that to regulates the size of the abstract features."""
    if hparams.prioritized_buffer:
        random_states_1 = replay_buffer.sample(hparams.batch_size)[1]
    else:
        random_states_1 = replay_buffer.sample(hparams.batch_size)[0]

    random_states_1 = agent.encode(torch.as_tensor(random_states_1, device=device))
    ld2 = torch.clamp(torch.max(torch.pow(random_states_1, 2)) - 1.0, 0.0, 100.0)

    return ld2


def compute_interp_loss(
    agent, encoded_batch, interp_vector, hparams=None, device="cpu"
):
    encoded_states, *_ = encoded_batch
    actions = torch.tensor([0] * hparams.batch_size, device=device)
    predicted_next_states = agent.compute_transition(encoded_states, actions)
    transition = predicted_next_states - encoded_states

    return -(nn.CosineSimilarity()(transition, interp_vector)).sum()
