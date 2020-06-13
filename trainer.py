import pytorch_lightning as pl
from crar.crar import CRARLightning
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
    # logger = WandbLogger(name=hparams.logger_dir.split("/")[1])
    # logger.watch(model, log="all", log_freq=10)

    grad_clip_norm = 0
    if "grad_clip_norm" in hparams:
        grad_clip_norm = hparams.optim.grad_clip_norm

    trainer = pl.Trainer(
        gpus=1,
        # logger=logger,
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
    # for _ in range(10):
    main(config)
