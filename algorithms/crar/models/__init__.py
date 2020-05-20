import torch
from .qnet import QLearner, QNetwork
from .encoder import Encoder, SimpleEncoder
from .reward_predictor import RewardPredictor
from .transition_predictor import TransitionPredictor
from .make_crar_agent import (
    make_encoder,
    make_qnet,
    make_transition_predictor,
    make_reward_predictor,
)


def synchronize_target_model(
    current_model: torch.nn.Module, target_model: torch.nn.Module
):
    target_model.load_state_dict(current_model.state_dict())
