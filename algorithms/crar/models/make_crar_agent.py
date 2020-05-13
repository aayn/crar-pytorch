import yaml
import torch
import torch.nn as nn
from .encoder import Encoder
from .qnet import QNetwork
from .transition_predictor import TransitionPredictor
from .reward_predictor import RewardPredictor

NN_MAP = {
    "Conv2d": nn.Conv2d,
    "MaxPool2d": nn.MaxPool2d,
    "tanh": nn.Tanh,
    "Linear": nn.Linear,
    "relu": nn.ReLU,
}


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
    print(fc_config)
    fc = []
    for i, layer in enumerate(fc_config):
        if layer[0] == "Linear":
            if layer[1] == "auto":
                fc.append(NN_MAP[layer[0]](input_dim, layer[2]))
            elif layer[2] == "auto":
                fc.append(NN_MAP[layer[0]](layer[1], out_dim))
            else:
                fc.append(NN_MAP[layer[0]](layer[1], layer[2]))
        else:
            fc.append(NN_MAP[layer]())

    return nn.Sequential(*fc)


def make_transition_predictor(abstract_dim, num_actions):
    with open("models/network.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    tp_config = config["trans-pred"]
    fc = make_fc(abstract_dim + num_actions, abstract_dim, tp_config["fc"])
    transition_predictor = TransitionPredictor(abstract_dim, num_actions, fc)
    return transition_predictor


def make_reward_predictor(abstract_dim, num_actions):
    with open("models/network.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    rp_config = config["reward-pred"]
    fc = make_fc(abstract_dim + num_actions, abstract_dim, rp_config["fc"])
    reward_predictor = RewardPredictor(abstract_dim, num_actions, fc)
    return reward_predictor


def make_encoder(input_shape, abstract_dim, device) -> Encoder:
    with open("models/network.yaml") as f:
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
    with open("models/network.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    qnet_config = config["qnet"]

    convs, feature_size = None, input_dim
    if qnet_config["convs"] is not None:
        convs = make_convs(input_dim, qnet_config["convs"])
        feature_size = compute_feature_size(input_dim, convs)
    fc = make_fc(feature_size, num_actions, qnet_config["fc"])

    qnet = QNetwork(input_dim, num_actions, convs, fc, device)
    return qnet


# def load_network(device, input_shape=(1, 48, 48), abstract_dim=2):
#     with open("network.yaml") as f:
#         data = yaml.load(f, Loader=yaml.FullLoader)
#     encoder = make_encoder(input_shape, abstract_dim, data["encoder"], device)

#     return encoder


# if __name__ == "__main__":
#     load_network()
