"""Measure Mean-Teacher pseudo-label scores on unlabeled weak views.

This is a diagnostic only: it loads a saved checkpoint, uses the configured
unlabeled data pipeline, and reports counts before and after the configured
paper threshold. It does not change the model or training configuration.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings
from mmdet.models import build_detector
from ssod.datasets import build_dataset
from ssod.models.utils import filter_invalid
from ssod.utils import patch_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="single-GPU or paper Mean-Teacher config")
    parser.add_argument("checkpoint", help="saved model checkpoint")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", help="optional JSON report path")
    return parser.parse_args()


def unwrap(value):
    return getattr(value, "data", value)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get("work_dir") is None:
        cfg.work_dir = (
            os.path.dirname(os.path.abspath(args.output))
            if args.output
            else os.path.abspath("./work_dirs/pseudo_label_diagnostic")
        )
    cfg = patch_config(cfg)
    if cfg.model.get("type") != "SingleStageMeanTeacher":
        raise ValueError(
            "This diagnostic expects the single-stage Mean-Teacher wrapper; "
            f"got {cfg.model.get('type')!r}"
        )

    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(args.device).eval()
    teacher = model.teacher

    dataset_cfg = cfg.data.train.unsup.copy()
    dataset = build_dataset(dataset_cfg)
    count = min(args.num_images, len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), count)
    threshold = float(model.train_cfg.pseudo_label_initial_score_thr)
    min_size = float(model.train_cfg.min_pseduo_box_size)
    rows = []

    with torch.no_grad():
        for index in indices:
            branches = dataset[index]
            teacher_sample = branches[0]
            image = unwrap(teacher_sample["img"])
            if not torch.is_tensor(image):
                raise TypeError(f"Expected image tensor, got {type(image)!r}")
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = image.to(args.device)

            meta = unwrap(teacher_sample["img_metas"])
            if isinstance(meta, (tuple, list)):
                meta = meta[0]

            features = teacher.extract_feat(image)
            detections = teacher.bbox_head.simple_test_bboxes(
                features, [meta], rescale=False
            )
            boxes, labels = detections[0]
            scores = boxes[:, -1]
            filtered_boxes, filtered_labels, _ = filter_invalid(
                boxes,
                labels,
                scores,
                thr=threshold,
                min_size=min_size,
            )
            rows.append(
                {
                    "dataset_index": int(index),
                    "filename": meta.get("filename"),
                    "detections_after_test_nms": int(boxes.shape[0]),
                    "detections_ge_threshold": int((scores > threshold).sum()),
                    "pseudo_boxes_after_filter": int(filtered_boxes.shape[0]),
                    "max_score": float(scores.max().item()) if scores.numel() else 0.0,
                }
            )

    report = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "threshold": threshold,
        "num_images": len(rows),
        "images_with_pseudo_boxes": sum(r["pseudo_boxes_after_filter"] > 0 for r in rows),
        "mean_pseudo_boxes_per_image": (
            sum(r["pseudo_boxes_after_filter"] for r in rows) / max(len(rows), 1)
        ),
        "mean_max_score": sum(r["max_score"] for r in rows) / max(len(rows), 1),
        "images": rows,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
