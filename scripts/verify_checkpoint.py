#!/usr/bin/env python3
"""Check teacher/student storage and state divergence in a saved checkpoint."""

import argparse
import torch
from mmcv import Config
from mmdet.models import build_detector
from ssod.utils import patch_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.work_dir = "./work_dirs/checkpoint_check"
    cfg = patch_config(cfg)
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    teacher_parameters = dict(model.teacher.named_parameters())
    student_parameters = dict(model.student.named_parameters())
    if not teacher_parameters or teacher_parameters.keys() != student_parameters.keys():
        raise SystemExit("teacher/student parameter names differ")
    shared = [
        name for name in teacher_parameters
        if teacher_parameters[name].data_ptr() == student_parameters[name].data_ptr()
    ]
    if shared:
        raise SystemExit(f"teacher/student share parameter storage: {shared[:3]}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    nonfinite = [
        name
        for name, value in state.items()
        if torch.is_tensor(value)
        and (value.is_floating_point() or value.is_complex())
        and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise SystemExit(f"checkpoint contains non-finite tensors: {nonfinite[:5]}")
    differing = []
    for name, student_value in state.items():
        if name.startswith("student."):
            teacher_name = "teacher." + name[len("student."):]
            teacher_value = state.get(teacher_name)
            if teacher_value is not None and student_value.shape == teacher_value.shape:
                if not torch.equal(student_value, teacher_value):
                    differing.append(name)
    if not differing:
        raise SystemExit("teacher/student checkpoint weights are identical; EMA update not evidenced")
    print(f"teacher/student storage is separate; {len(differing)} checkpoint parameter tensors differ")


if __name__ == "__main__":
    main()
