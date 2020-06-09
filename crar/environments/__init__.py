from .simple_maze import SimpleMaze
from gym.envs.registration import register
from .wrappers import wrap_pytorch, wrap_deepmind, make_atari, TimeLimit


register(
    id="SimpleMaze-v0",
    entry_point="crar.environments:SimpleMaze",
    reward_threshold=0.0,
    kwargs={"id": "SimpleMaze-v0", "higher_dim_obs": True, "has_goal": False},
)


register(
    id="SimpleMaze-v1",
    entry_point="crar.environments:SimpleMaze",
    kwargs={"id": "SimpleMaze-v1", "higher_dim_obs": True, "has_goal": True},
)
