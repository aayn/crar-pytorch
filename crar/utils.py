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
    eps = max(eps_end, eps_start - (current_frame / eps_last_frame))
    return eps
