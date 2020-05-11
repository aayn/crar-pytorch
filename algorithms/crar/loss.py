import torch
import torch.optim as optim
from model import QLearner, SimpleEncoder, Encoder, RewardPredictor
from utils import ReplayBuffer
from typing import Union


def mean_squared_error_p(y_true, y_pred):
    """ Modified mean square error that clips
    """
    return torch.clamp(
        torch.max(torch.pow(torch.y_pred - y_true, 2), dim=-1) - 1, 0.0, 100.0
    )


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

    q_values = current_model.get_value(states)
    next_q_values = current_model.get_value(next_states)
    target_next_q_values = target_model.get_value(next_states)

    q_value = torch.gather(q_values, 1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = torch.gather(
        target_next_q_values, 1, torch.max(next_q_values, 1)[1].unsqueeze(1)
    ).squeeze(1)
    expected_q_value = torch.as_tensor(
        rewards + gamma * next_q_value * (1 - done.int()), device=device
    )

    loss = (q_value - expected_q_value).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def crar_loss_optim(
    batch_size: int,
    gamma: float,
    encoder: Union[SimpleEncoder, Encoder],
    reward_predictor: RewardPredictor,
    current_model: QLearner,
    target_model: QLearner,
    optimizer: optim.Optimizer,
    encoder_optimizer: optim.Optimizer,
    reward_pred_optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    device,
):
    states, actions, rewards, next_states, done = map(
        lambda x: torch.tensor(x, device=device), replay_buffer.sample(batch_size)
    )
    states = encoder(states)
    next_states = encoder(next_states)

    state_action = torch.cat((states, actions.unsqueeze(1).float()), 1)
    predicted_rewards = reward_predictor(state_action)

    q_values = current_model.get_value(states)
    next_q_values = current_model.get_value(next_states)
    target_next_q_values = target_model.get_value(next_states)

    q_value = torch.gather(q_values, 1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = torch.gather(
        target_next_q_values, 1, torch.max(next_q_values, 1)[1].unsqueeze(1)
    ).squeeze(1)
    expected_q_value = torch.as_tensor(
        rewards + gamma * next_q_value * (1 - done.int()), device=device
    )

    two_states, *_ = map(
        lambda x: torch.tensor(x, device=device), replay_buffer.sample(2)
    )
    two_states = encoder(two_states)

    Cd = 5
    beta = 0.2
    ld1 = torch.exp(-Cd * torch.norm(two_states[0] - two_states[1]))
    ld1_ = torch.exp(-Cd * torch.norm(states[0] - next_states[0]))
    ld2 = torch.max(torch.norm(two_states[0], p=float("inf")) - 1, 0)[0]
    representation_loss = ld1 + beta * ld1_ + ld2

    double_q_loss = (q_value - expected_q_value).pow(2).mean()
    reward_loss = (predicted_rewards - rewards).pow(2).mean()
    loss = double_q_loss + reward_loss + representation_loss

    optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    reward_pred_optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    encoder_optimizer.step()
    reward_pred_optimizer.step()

    return double_q_loss, reward_loss, representation_loss, loss
