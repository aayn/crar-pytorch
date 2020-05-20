import pytorch_lightning as pl
from crar_lightning import CRARLightning
import numpy as np
import torch
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import os
import yaml
from box import Box
from pytorch_lightning.loggers import WandbLogger


def main(hparams):
    # TODO: Check that all view, reshape, transpose are used correctly
    model = CRARLightning(hparams)

    # logger = TensorBoardLogger(save_dir=os.getcwd(), name=hparams.logger_dir)
    logger = WandbLogger(name="test_plots15_interp")
    logger.watch(model, log="all", log_freq=100)

    grad_clip_norm = 0
    try:
        grad_clip_norm = hparams.optim.grad_clip_norm
    except KeyError:
        pass

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        distributed_backend="dp",
        max_epochs=hparams.max_epochs,
        early_stop_callback=False,
        gradient_clip_val=grad_clip_norm,
        benchmark=True,
        # auto_lr_find=True
        # val_check_interval=100,
        # log_gpu_memory="all",
    )

    trainer.fit(model)


if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="PongNoFrameskip-v4", help="gym environment tag"
    )
    args, _ = parser.parse_known_args()
    with open("config.yaml") as f:
        config = Box(yaml.load(f, Loader=yaml.FullLoader)[args.env])

    print(config)
    main(config)

    # parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    # parser.add_argument(
    #     "--sync_rate",
    #     type=int,
    #     default=1000,
    #     help="how many frames do we update the target network",
    # )
    # parser.add_argument(
    #     "--replay_size", type=int, default=100000, help="capacity of the replay buffer"
    # )
    # parser.add_argument(
    #     "--warm_start_size",
    #     type=int,
    #     default=10000,
    #     help="how many samples do we use to fill our buffer at the start of training",
    # )
    # parser.add_argument(
    #     "--eps_last_frame",
    #     type=int,
    #     default=250000,
    #     help="what frame should epsilon stop decaying",
    # )
    # parser.add_argument(
    #     "--eps_start", type=float, default=1.0, help="starting value of epsilon"
    # )
    # parser.add_argument(
    #     "--eps_end", type=float, default=0.01, help="final value of epsilon"
    # )
    # parser.add_argument(
    #     "--episode_length", type=int, default=2000, help="max length of an episode"
    # )
    # parser.add_argument(
    #     "--max_episode_reward",
    #     type=int,
    #     default=21,
    #     help="max episode reward in the environment",
    # )
    # parser.add_argument(
    #     "--warm_start_steps",
    #     type=int,
    #     default=50000,
    #     help="max episode reward in the environment",
    # )
    # parser.add_argument(
    #     "--is_atari", action="store_true", help="are you running an atari environment?"
    # )
