import inspect
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from .encoder import Encoder
from .qnet import QNetwork
from .transition_predictor import TransitionPredictor
from .scalar_predictor import ScalarPredictor
from crar.utils import nonthrowing_issubclass

NN_MAP = {
    k: v for k, v in inspect.getmembers(nn) if nonthrowing_issubclass(v, nn.Module)
}
HERE = Path(__file__).parent


def compute_feature_size(input_shape, convs):
    return convs(torch.zeros(1, *input_shape)).view(1, -1).size(1)


def make_convs(input_shape, conv_config):
    convs = []
    for i, layer in enumerate(conv_config):
        if layer[0] == "Conv2d":
            if layer[1] == "auto":
                convs.append(NN_MAP[layer[0]](input_shape[0], layer[2], **layer[3]))
            else:
                convs.append(NN_MAP[layer[0]](layer[1], layer[2], **layer[3]))
        elif layer[0] == "MaxPool2d":
            convs.append(NN_MAP[layer[0]](**layer[1]))
        else:
            convs.append(NN_MAP[layer]())

    return nn.Sequential(*convs)


def make_fc(input_dim, out_dim, fc_config):
    fc = []
    for i, layer in enumerate(fc_config):
        if layer[0] == "Linear":
            if layer[1] == "auto" and layer[2] == "auto":
                fc.append(NN_MAP[layer[0]](input_dim, out_dim))
            elif layer[1] == "auto":
                fc.append(NN_MAP[layer[0]](input_dim, layer[2]))
            elif layer[2] == "auto":
                fc.append(NN_MAP[layer[0]](layer[1], out_dim))
            else:
                fc.append(NN_MAP[layer[0]](layer[1], layer[2]))
        else:
            fc.append(NN_MAP[layer]())

    return nn.Sequential(*fc)


def make_transition_predictor(abstract_dim, num_actions):
    with open(HERE / "network.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    tp_config = config["trans-pred"]
    fc = make_fc(abstract_dim + 1, abstract_dim, tp_config["fc"])
    transition_predictor = TransitionPredictor(abstract_dim, num_actions, fc)
    return transition_predictor


def make_scalar_predictor(config_name, abstract_dim, num_actions):
    with open(HERE / "network.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    rp_config = config[config_name]
    fc = make_fc(abstract_dim + 1, abstract_dim, rp_config["fc"])

    scalar_predictor = ScalarPredictor(abstract_dim, num_actions, fc)
    return scalar_predictor


def make_reward_predictor(abstract_dim, num_actions):
    reward_predictor = make_scalar_predictor("reward-pred", abstract_dim, num_actions)
    return reward_predictor


def make_discount_predictor(abstract_dim, num_actions):
    discount_predictor = make_scalar_predictor(
        "discount-pred", abstract_dim, num_actions
    )
    return discount_predictor


def make_encoder(input_shape, abstract_dim, device) -> Encoder:
    with open(HERE / "network.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    encoder_config = config["encoder"]

    convs, feature_size = None, input_shape[0]
    if encoder_config["convs"] is not None:
        convs = make_convs(input_shape, encoder_config["convs"])
        feature_size = compute_feature_size(input_shape, convs)
    fc = make_fc(feature_size, abstract_dim, encoder_config["fc"])

    encoder = Encoder(input_shape, device, fc, convs)
    return encoder


def make_qnet(input_dim, num_actions, device) -> QNetwork:
    with open(HERE / "network.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    qnet_config = config["qnet"]

    convs, feature_size = None, input_dim
    if qnet_config["convs"] is not None:
        convs = make_convs(input_dim, qnet_config["convs"])
        feature_size = compute_feature_size(input_dim, convs)
    fc = make_fc(feature_size, num_actions, qnet_config["fc"])

    qnet = QNetwork(input_dim, num_actions, convs, fc, device)
    return qnet


if __name__ == "__main__":
    # Utility to explore and test the YAML network file
    def load_network(device, input_shape=(1, 48, 48), abstract_dim=2):
        with open("network.yaml") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        encoder = make_encoder(input_shape, abstract_dim, data["encoder"], device)

        return encoder

    load_network()
