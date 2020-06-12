import abc
from collections import namedtuple, deque
from typing import Tuple, Union, List, NewType, Any
import numpy as np
from torch.utils.data.dataset import IterableDataset
import bisect


Experience = namedtuple(
    "Experience",
    field_names=["observation", "action", "reward", "done", "next_observation"],
)

# We need to modify the priority (abs_td_error) in-place and hence don't use a namedtuple
TDExperience = NewType("TDExperience", List[Any])

# TDExperience = namedtuple(
#     "TDExperience",
#     field_names=[
#         "abs_td_error",
#         "observation",
#         "action",
#         "reward",
#         "done",
#         "next_observation",
#     ],
# )


class ReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def push(self, experience: Union[Experience, TDExperience]) -> None:
        ...

    @abc.abstractmethod
    def sample(self, batch_size: int, replace: bool) -> Tuple:
        ...


class UniformReplayBuffer(ReplayBuffer):
    """Replay buffer to store past experience.

    Args:
        capacity: size of buffer
    """

    def __init__(self, capacity: Union[int, None]) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, experience: Experience) -> None:
        self.buffer.appendleft(experience)

    def sample(self, batch_size: int, replace=False) -> Tuple:
        if batch_size > len(self) and not replace:
            raise ValueError("Sample batch size is greater than size of buffer.")
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


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: Union[int, None] = None,
        sort_interval=10000,
        num_segments=32,
        prioritization_alpha=0.7,
    ) -> None:
        self.capacity = capacity
        self.buffer = []
        self.sort_interval = sort_interval
        self.interval_count = 0
        self.prioritization_alpha = prioritization_alpha

        self.num_segments = num_segments

        self.segment_indices = None
        # self._compute_segment_indices()
        self.segment_probs = None
        self.recent_sample_indices = []

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, experience: TDExperience) -> None:
        self.buffer.append(experience)
        self.interval_count += 1
        # Amortized sorting, as described in:
        # https://arxiv.org/abs/1511.06581
        if self.interval_count >= self.sort_interval:
            self.buffer.sort(key=lambda e: e[0])
            self.interval_count = 0
            # TODO: Maybe have a separate interval for the following updates
            self._compute_segment_indices()
            self._compute_segment_probs()

        if self.capacity is not None and len(self.buffer) > self.capacity:
            self.buffer.pop()

    def sample(self, batch_size: int, replace=False) -> Tuple:
        if batch_size > len(self) and not replace:
            raise ValueError("Sample batch size is greater than size of buffer.")
        if self.segment_indices is None:
            self._compute_segment_indices()
            self._compute_segment_probs()

        self.interval_count += batch_size

        segment_samples = np.random.choice(
            self.segment_indices, batch_size, replace=True, p=self.segment_probs
        )
        self.recent_sample_indices = []
        for i in segment_samples:
            ri = bisect.bisect(self.segment_indices, i)
            li = ri - 1
            try:
                lbound, rbound = self.segment_indices[li], self.segment_indices[ri]
            except IndexError:
                lbound, rbound = self.segment_indices[li], len(self.buffer)

            current_idx = np.random.randint(lbound, rbound)
            self.recent_sample_indices.append(current_idx)

        priorities, states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in self.recent_sample_indices]
        )
        return (
            np.array(priorities),
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states),
        )

    def update_priorities(self, td_error):
        for i, si in enumerate(self.recent_sample_indices):
            self.buffer[si][0] = -abs(td_error[i].item())
        self.recent_sample_indices = []

    def _compute_segment_indices(self):
        if self.num_segments > len(self.buffer):
            raise ValueError("Number of segments > buffer size.")
        self.segment_indices = []
        idx = 0
        increment = len(self.buffer) // self.num_segments
        mod = len(self.buffer) % self.num_segments
        for i in range(self.num_segments):
            self.segment_indices.append(idx)
            idx += increment + int(i < mod)

    def _compute_segment_probs(self):
        if self.segment_indices is None:
            raise ValueError("Segments not computed.")
        self.segment_probs = list(
            map(
                lambda i: pow(1 / (1 + i), self.prioritization_alpha),
                self.segment_indices,
            )
        )
        total_prob = sum(self.segment_probs)
        self.segment_probs = list(map(lambda pi: pi / total_prob, self.segment_probs))


# Inspired, in part, by:
#     https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31


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
