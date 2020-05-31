import numpy as np
import random
import torch
from functools import wraps
from time import time
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


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        ret = f(*args, **kwargs)
        end = time()
        print(f"Func {f.__name__} took {end - start:.5f} sec")
        return ret

    return wrapper
