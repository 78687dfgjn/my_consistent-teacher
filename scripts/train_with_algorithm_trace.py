#!/usr/bin/env python3
"""Run tools/train.py while counting real FAM-3D and ASA method calls."""

import runpy
import os
import csv
import inspect
import sys
import atexit
from pathlib import Path

from amp_optimizer_trace import install_amp_optimizer_trace
from ssod.core.bbox.assigners.dynamic_assigner import DynamicSoftLabelAssigner
from ssod.models.consistent_teacher import ConsistentTeacher
from ssod.models.dense_heads.fam3d import FAM3DHead

calls = {"fam3d_forward": 0, "dynamic_assign": 0, "gmm_policy": 0}
fam3d_forward = FAM3DHead.forward
dynamic_assign = DynamicSoftLabelAssigner.assign
gmm_policy = ConsistentTeacher.gmm_policy
trace_file = os.environ.get("CT_ALGORITHM_TRACE_FILE")
threshold_trace_file = os.environ.get("CT_GMM_THRESHOLD_TRACE_FILE")
threshold_trace_interval = int(os.environ.get("CT_GMM_THRESHOLD_TRACE_INTERVAL", "1000"))
if threshold_trace_interval <= 0:
    raise ValueError("CT_GMM_THRESHOLD_TRACE_INTERVAL must be a positive integer")
threshold_trace_seen = set()
threshold_writer = None
threshold_stream = None

if threshold_trace_file:
    threshold_path = Path(threshold_trace_file)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    if threshold_path.exists():
        with threshold_path.open(newline="") as existing:
            for row in csv.DictReader(existing):
                try:
                    threshold_trace_seen.add((int(row["iteration"]), int(row["class_id"])))
                except (KeyError, TypeError, ValueError):
                    continue
    is_new_trace = not threshold_path.exists() or threshold_path.stat().st_size == 0
    threshold_stream = threshold_path.open("a", newline="")
    threshold_writer = csv.writer(threshold_stream)
    if is_new_trace:
        threshold_writer.writerow(["iteration", "class_id", "threshold", "queue_scores"])
        threshold_stream.flush()


def traced_fam3d_forward(self, *args, **kwargs):
    calls["fam3d_forward"] += 1
    return fam3d_forward(self, *args, **kwargs)


def traced_dynamic_assign(self, *args, **kwargs):
    calls["dynamic_assign"] += 1
    return dynamic_assign(self, *args, **kwargs)


def traced_gmm_policy(self, *args, **kwargs):
    calls["gmm_policy"] += 1
    threshold = gmm_policy(self, *args, **kwargs)
    if threshold_writer is not None:
        caller_locals = inspect.currentframe().f_back.f_locals
        label = caller_locals.get("label")
        iteration = int(getattr(self, "iter", -1))
        if label is not None and iteration > 0 and iteration % threshold_trace_interval == 0:
            key = (iteration, int(label))
            if key not in threshold_trace_seen:
                threshold_trace_seen.add(key)
                class_queue = self.scores[int(label)]
                queue_score_count = int((class_queue > 0).sum().item())
                threshold_writer.writerow(
                    [iteration, int(label), f"{float(threshold):.8f}", queue_score_count]
                )
                threshold_stream.flush()
    return threshold


def write_algorithm_trace():
    trace = (
        "CT_ALGORITHM_TRACE "
        f"FAM3D.forward={calls['fam3d_forward']} "
        f"DynamicSoftLabelAssigner.assign={calls['dynamic_assign']} "
        f"GMM.policy={calls['gmm_policy']}"
    )
    print(trace)
    if trace_file:
        with open(trace_file, "a") as output:
            output.write(trace + "\n")
    if threshold_stream is not None:
        threshold_stream.close()


def main():
    install_amp_optimizer_trace()
    FAM3DHead.forward = traced_fam3d_forward
    DynamicSoftLabelAssigner.assign = traced_dynamic_assign
    ConsistentTeacher.gmm_policy = traced_gmm_policy
    atexit.register(write_algorithm_trace)
    sys.argv = ["tools/train.py", *sys.argv[1:]]
    runpy.run_path("tools/train.py", run_name="__main__")
    if not all(calls.values()):
        raise SystemExit("FAM-3D or DynamicSoftLabelAssigner was not called")


if __name__ == "__main__":
    main()
