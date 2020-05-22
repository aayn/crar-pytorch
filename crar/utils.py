import numpy as np
import random
import torch
import gym
import torch.optim as optim
from itertools import accumulate
import torch.multiprocessing as mp
from collections import deque


def nonthrowing_issubclass(cl1, cl2) -> bool:
    try:
        return issubclass(cl1, cl2)
    except TypeError:
        pass
    return False


def compute_eps(current_frame, eps_start=1.0, eps_end=0.0, eps_last_frame=float("inf")):
    eps = eps_end + (eps_start - eps_end) * np.exp(
        -1.0 * current_frame / eps_last_frame
    )

    # max(eps_end, eps_start - (current_frame / eps_last_frame))
    return eps


# class ReplayBuffer:
#     def __init__(self, buffer_size):
#         self.buffer = deque(maxlen=buffer_size)

#     def push(self, state, action, reward, next_state, done):
#         state = np.expand_dims(state, 0)
#         next_state = np.expand_dims(next_state, 0)

#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         states, actions, rewards, next_states, done = zip(
#             *random.sample(self.buffer, batch_size)
#         )

#         return (
#             np.concatenate(states),
#             actions,
#             rewards,
#             np.concatenate(next_states),
#             done,
#         )

#     def __len__(self):
#         return len(self.buffer)


# # NOTE: This version is faster
# # @timeit
# def reward_to_go(rewards):
#     rtg = list(reversed(list(accumulate(reversed(rewards)))))
#     return rtg


# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""

#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__
