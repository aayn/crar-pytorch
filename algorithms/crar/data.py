# Credits: https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=NWvMLBDySQI5

from collections import namedtuple, deque
from typing import Tuple
import numpy as np
from torch.utils.data.dataset import IterableDataset

Experience = namedtuple(
    "Experience", field_names=["observation", "action", "reward", "done", "next_state"]
)


class ReplayBuffer:
    """Replay buffer to learn from past experience.

    Args:
        capacity: size of buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
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


class ExperienceDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 128):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, next_states = self.buffer.sample(
            self.sample_size
        )
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], next_states[i]
