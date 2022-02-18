import argparse
import numpy as np

import yaml
from easydict import EasyDict as edict
import torch
import torch.nn as nn

import os
from darwinai.torch.builder import build_model  # , BlockSpec, BuildMetrics
from darwinai.builder import BlockSpec
from darwin.enums.enums import BuildMetrics
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from architectures.DarwinNet_groups4 import DarwinNetV2

gsbuild_config = {
    "blockspecs": [
        BlockSpec(channels=40, depth=3),
        BlockSpec(channels=84, depth=4),
        BlockSpec(channels=176, depth=7),
        BlockSpec(channels=372, depth=3),
    ]
}

INPUT_SHAPE = [224, 224, 3]


def make_model(blockspecs):
    return DarwinNetV2(blockspecs, INPUT_SHAPE, 1000)


def main(checkpoint_path):
    build_metric = BuildMetrics.FLOPS
    target_ratio = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baseline_pretrained_model = make_model(gsbuild_config["blockspecs"])
    baseline_pretrained_model.to(device)
    baseline_pretrained_model.load_state_dict(
        torch.load(checkpoint_path, map_location=torch.device("cpu")), strict=True
    )

    model = build_model(
        model_fn=make_model,
        initial_blockspecs=gsbuild_config["blockspecs"],
        input_shape=[1, INPUT_SHAPE[2], INPUT_SHAPE[0], INPUT_SHAPE[1]],
        target_ratio=target_ratio,
        build_metric=build_metric,
        pretrained_model=baseline_pretrained_model,
    )

    print(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.set_defaults(reshuffle=False, augment=False)
    args = parser.parse_args()
    main(args.checkpoint)
