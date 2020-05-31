from .simple_maze import SimpleMaze
from gym.envs.registration import register


register(
    id="SimpleMaze-v0",
    entry_point="crar.environments:SimpleMaze",
    reward_threshold=0.0,
    kwargs={"higher_dim_obs": True, "has_goal": False},
)


register(
    id="SimpleMaze-v1",
    entry_point="crar.environments:SimpleMaze",
    kwargs={"higher_dim_obs": True, "has_goal": True},
)
