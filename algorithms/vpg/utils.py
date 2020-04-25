import numpy as np
import torch
import gym
from model import Actor, Critic
import torch.optim as optim
from itertools import accumulate
import torch.multiprocessing as mp


# NOTE: This version is faster
# @timeit
def reward_to_go(rewards):
    rtg = list(reversed(list(accumulate(reversed(rewards)))))
    return rtg


# @timeit
def reward_to_go_2(rewards):
    rtg = [0] * len(rewards)
    for i, r in enumerate(reversed(rewards), 1):
        rtg[-i] = r + rtg[-i + 1]
    return rtg


def critic_iterate(
    pid, critic: Critic, critic_iters: int, batch_obs, returns, critic_optimizer
):
    for _ in range(critic_iters):
        critic_loss = critic.get_loss(batch_obs, returns)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()


def train_one_epoch(
    batch_size: int,
    actor: Actor,
    critic: Critic,
    env: gym.Env,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    critic_iters: int,
):
    batch_obs = []
    batch_acts = []
    batch_rews = []
    weights = []

    ep_rews = []
    critic_preds = []
    obs = env.reset()
    while True:
        action = actor.get_action(obs)
        batch_obs.append(obs.copy())
        batch_acts.append(action)
        obs, reward, done, _ = env.step(action)
        critic_val = critic.get_state_value(obs)
        ep_rews.append(reward)
        critic_preds.append(critic_val)
        if done:
            # print(ep_rews)
            batch_rews.append(sum(ep_rews))
            ep_rews = reward_to_go(ep_rews)

            weights += ep_rews
            if len(batch_obs) > batch_size:
                break
            ep_rews = []
            obs = env.reset()

    returns = weights[:]
    mp.spawn(
        fn=critic_iterate,
        args=(critic, critic_iters, batch_obs, returns, critic_optimizer),
    )
    weights = list(map(lambda w, c: w - c, weights, critic_preds))

    act_loss = actor.get_loss(batch_obs, batch_acts, weights)
    actor_optimizer.zero_grad()
    act_loss.backward()
    actor_optimizer.step()

    return np.mean(batch_rews)


def play_on_env(env: gym.Env, actor: Actor, times: int):

    obs = env.reset()
    env.render()
    done = False
    for _ in range(times):
        while not done:
            action = actor.get_action(obs)
            obs, _, done, _ = env.step(action)
            env.render()

        if done:
            obs = env.reset()
            done = False
