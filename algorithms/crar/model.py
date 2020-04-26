import abc
import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from itertools import accumulate
from perf import timeit

USE_CUDA = torch.cuda.is_available()
device = "cuda" if USE_CUDA else "cpu"
device = torch.device(device)


class Encoder(nn.Module):
    def __init__(self, input):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv2d())


if __name__ == "__main__":
    pass
