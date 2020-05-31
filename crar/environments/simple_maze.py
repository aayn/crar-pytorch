"""
Replication of
    https://github.com/VinF/deer/blob/master/examples/test_CRAR/simple_maze_env.py.
"""
import copy
import gym
import numpy as np
import matplotlib

# matplotlib.use('agg')
matplotlib.use("qt5agg")
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.patches import Circle, Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker


class SimpleMaze(gym.Env):
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Space((1, 48, 48), float)

    def __init__(self, **kwargs):
        self._mode = -1
        self._size_maze = 8
        self._higher_dim_obs = kwargs["higher_dim_obs"]
        self._has_goal = kwargs["has_goal"]
        self.create_map()
        self.intern_dim = 2
        # Need to set this in order for it to work with OpenAI Gym
        self._gym_disable_underscore_compat = True

    def create_map(self):
        self._map = np.zeros((self._size_maze, self._size_maze))
        self._map[-1, :] = 1
        self._map[0, :] = 1
        self._map[:, 0] = 1
        self._map[:, -1] = 1
        self._map[:, self._size_maze // 2] = 1
        self._map[self._size_maze // 2, self._size_maze // 2] = 0
        self._pos_agent = [2, 2]
        if self._has_goal:
            self._pos_goal = [self._size_maze - 2, self._size_maze - 2]

    def reset(self):
        self.create_map()
        self._map[self._size_maze // 2, self._size_maze // 2] = 0

        # Setting the starting position of the agent
        self._pos_agent = [self._size_maze // 2, self._size_maze // 2]
        return self.observe()

    def step(self, action):
        self._cur_action = action
        if action == 0:
            if self._map[self._pos_agent[0] - 1, self._pos_agent[1]] == 0:
                self._pos_agent[0] = self._pos_agent[0] - 1
        elif action == 1:
            if self._map[self._pos_agent[0] + 1, self._pos_agent[1]] == 0:
                self._pos_agent[0] = self._pos_agent[0] + 1
        elif action == 2:
            if self._map[self._pos_agent[0], self._pos_agent[1] - 1] == 0:
                self._pos_agent[1] = self._pos_agent[1] - 1
        elif action == 3:
            if self._map[self._pos_agent[0], self._pos_agent[1] + 1] == 0:
                self._pos_agent[1] = self._pos_agent[1] + 1

        done = False
        if self._has_goal:
            self.reward = -1.0
            if self._pos_agent == self._pos_goal:
                self.reward = 100.0
                done = True
        else:
            # There is no reward in SimpleMaze-v0
            self.reward = 0
        return self.observe(), self.reward, done, None

    def render(self):
        return self.observe()

    def observe(self):
        obs = copy.deepcopy(self._map)

        obs[self._pos_agent[0], self._pos_agent[1]] = 0.5
        if self._higher_dim_obs is True:
            "self._pos_agent"
            self._pos_agent
            obs = self.get_higher_dim_obs([self._pos_agent], [self._pos_goal])
        return [obs]

    def get_higher_dim_obs(self, indices_agent, indices_reward):
        """
        Converts underlying simple maze to higher-dimensional observation.
        """
        obs = copy.deepcopy(self._map)
        obs = obs / 1.0
        obs = np.repeat(np.repeat(obs, 6, axis=0), 6, axis=1)
        # Agent representation
        agent_obs = np.zeros((6, 6))
        agent_obs[0, 2] = 0.5
        agent_obs[1, 0:5] = 0.6
        agent_obs[2, 1:4] = 0.6
        agent_obs[3, 1:4] = 0.6
        agent_obs[4, 1] = 0.6
        agent_obs[4, 3] = 0.6
        agent_obs[5, 0:2] = 0.6
        agent_obs[5, 3:5] = 0.6

        # Reward representation
        reward_obs = np.zeros((6, 6))
        if self._has_goal:
            reward_obs[:3, 1] = 0.95
            reward_obs[0, 1:4] = 0.95
            reward_obs[1, 1:4] = 0.95
            reward_obs[2, 1:4] = 0.95

        for i in indices_reward:
            obs[i[0] * 6 : (i[0] + 1) * 6 :, i[1] * 6 : (i[1] + 1) * 6] = reward_obs

        for i in indices_agent:
            obs[i[0] * 6 : (i[0] + 1) * 6 :, i[1] * 6 : (i[1] + 1) * 6] = agent_obs

        return obs

    def all_possible_inputs(self):
        """
        Returns an array of states with all possible agent positions.
        """
        all_possib_inp = []
        self.create_map()
        for y_a in range(self._size_maze):
            for x_a in range(self._size_maze):
                state = copy.deepcopy(self._map)
                state[self._size_maze // 2, self._size_maze // 2] = 0
                if state[x_a, y_a] == 0:
                    if self._higher_dim_obs is True:
                        all_possib_inp.append(
                            self.get_higher_dim_obs([[x_a, y_a]], [self._pos_goal])
                        )
                    else:
                        state[x_a, y_a] = 0.5
                        all_possib_inp.append(state)
        return np.array([all_possib_inp]).reshape(-1, 1, 48, 48)
