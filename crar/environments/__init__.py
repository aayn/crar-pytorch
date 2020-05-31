from .simple_maze import SimpleMaze
from gym.envs.registration import register

# TODO: Test this environment
# Perhaps v0 be one without reward and v1 is with
register(
    id="SimpleMaze-v0",
    entry_point="crar.environments:SimpleMaze",
    reward_threshold=0.0,
)
