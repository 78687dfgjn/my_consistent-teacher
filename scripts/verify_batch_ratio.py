#!/usr/bin/env python3
"""Verify that the configured sampler emits one labeled and four unlabeled."""

import argparse
import copy

from mmcv import Config
from ssod.datasets import build_dataset
from ssod.datasets.builder import build_sampler
from ssod.utils import patch_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.work_dir = "./work_dirs/batch_ratio_check"
    cfg = patch_config(cfg)
    batch_size = cfg.data.samples_per_gpu
    if batch_size != 5:
        raise SystemExit(f"expected 5 samples/GPU, got {batch_size}")
    dataset = build_dataset(cfg.data.train)
    sup_boundary = len(dataset.datasets[0])
    sampler = build_sampler(
        copy.deepcopy(cfg.data.sampler.train),
        dist=False,
        default_args=dict(dataset=dataset, samples_per_gpu=batch_size),
    )
    indices = iter(sampler)
    batches_checked = 0
    for _ in range(100):
        batch = [next(indices) for _ in range(batch_size)]
        n_sup = sum(index < sup_boundary for index in batch)
        n_unsup = batch_size - n_sup
        if (n_sup, n_unsup) != (1, 4):
            raise SystemExit(f"batch {batches_checked} has {n_sup} labeled / {n_unsup} unlabeled")
        batches_checked += 1
    print(f"verified {batches_checked} batches: each has 1 labeled + 4 unlabeled")


if __name__ == "__main__":
    main()
