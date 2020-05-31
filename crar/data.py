"""
Inspired by:
    https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31
"""
from collections import namedtuple, deque
from typing import Tuple, Union
import numpy as np
from torch.utils.data.dataset import IterableDataset

Experience = namedtuple(
    "Experience",
    field_names=["observation", "action", "reward", "done", "next_observation"],
)


class ReplayBuffer:
    """Replay buffer to store past experience.

    Args:
        capacity: size of buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, experience: Experience) -> None:
        self.buffer.appendleft(experience)

    def sample(self, batch_size: int, replace=False) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=replace)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states),
        )

    def most_recent(self, n: Union[None, int] = None):
        if n is None:
            return self.buffer[0]
        dq_copy, vals = self.buffer.copy(), []
        for _ in range(n):
            vals.append(dq_copy.popleft())

        return vals


class ExperienceDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 128, replace=False):
        self.buffer = buffer
        self.sample_size = sample_size
        self.replace = replace

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, next_states = self.buffer.sample(
            self.sample_size, self.replace
        )
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], next_states[i]
