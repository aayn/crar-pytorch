import torch
import torch.nn as nn
import argparse
from utils import train_one_epoch, play_on_env
from model import MLPActor
import gym


def train():
    pass


def play():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "--environment", type=str, default="CartPole-v0")
    parser.add_argument("--times", type=int, default=1)
    # parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    env = gym.make(args.env)

    USE_CUDA = torch.cuda.is_available()
    device = "cuda" if USE_CUDA else "cpu"
    device = torch.device(device)

    actor = MLPActor(
        env.observation_space.shape[0], 1, 32, nn.PReLU, env.action_space.n, device
    ).to(device)
    actor.load_state_dict(torch.load("saved_models/best_actor.pt"))
    play_on_env(env, actor, args.times)


def main():
    pass


if __name__ == "__main__":
    play()
