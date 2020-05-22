import torch
from .make_crar_agent import (
    make_encoder,
    make_qnet,
    make_transition_predictor,
    make_reward_predictor,
    make_discount_predictor,
)


def synchronize_target_model(
    current_model: torch.nn.Module, target_model: torch.nn.Module
):
    target_model.load_state_dict(current_model.state_dict())
