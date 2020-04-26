import torch
import torch.optim as optim
from model import QLearner
from utils import ReplayBuffer


def q_td_loss_optim(
    batch_size: int,
    gamma: float,
    model: QLearner,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    device,
):
    states, actions, rewards, next_states, done = map(
        lambda x: torch.tensor(x, device=device), replay_buffer.sample(batch_size)
    )

    q_values = model.get_value(states)
    next_q_values = model.get_value(next_states)

    q_value = torch.gather(q_values, 1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = torch.max(next_q_values, 1)[0]
    expected_q_value = torch.as_tensor(
        rewards + gamma * next_q_value * (1 - done.int())
    )

    loss = (q_value - expected_q_value).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def double_q_td_loss_optim(
    batch_size: int,
    gamma: float,
    current_model: QLearner,
    target_model: QLearner,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    device,
):
    states, actions, rewards, next_states, done = map(
        lambda x: torch.tensor(x, device=device), replay_buffer.sample(batch_size)
    )

    q_values = current_model.get_values(states)
    next_q_values = current_model.get_values(next_states)
    target_next_q_values = target_model.get_values(next_states)

    q_value = torch.gather(q_values, 1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = torch.gather(
        target_next_q_values, 1, torch.max(next_q_values, 1)
    ).squeeze(1)
    expected_q_value = torch.as_tensor(
        rewards + gamma * next_q_value * (1 - done.int())
    )

    loss = (q_value - expected_q_value).pow(2).mean()
    return loss
