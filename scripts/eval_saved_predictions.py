#!/usr/bin/env python3
"""Evaluate a saved MMDetection prediction pickle on the configured dataset."""

import argparse
import os

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from ssod.utils import patch_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("predictions")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.output_dir
    # Match tools/test.py's test loader: test_mode retains all 5,000 val
    # images, including the 48 images without annotations. When built directly
    # via build_dataset, CocoDataset otherwise applies its training-time empty
    # ground-truth filter and produces only 4,952 entries.
    cfg.data.test.test_mode = True
    cfg.data.test.filter_empty_gt = False
    cfg = patch_config(cfg)
    dataset = build_dataset(cfg.data.test)
    predictions = mmcv.load(args.predictions)
    metrics = dataset.evaluate(predictions, metric="bbox")
    os.makedirs(args.output_dir, exist_ok=True)
    mmcv.dump(
        {"config": args.config, "predictions": args.predictions, "metric": metrics},
        os.path.join(args.output_dir, "evaluation_metrics.json"),
    )
    print(metrics)


if __name__ == "__main__":
    main()
