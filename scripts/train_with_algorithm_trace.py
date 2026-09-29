#!/usr/bin/env python3
"""Run tools/train.py while counting real FAM-3D and ASA method calls."""

import runpy
import sys
import os

from ssod.core.bbox.assigners.dynamic_assigner import DynamicSoftLabelAssigner
from ssod.models.consistent_teacher import ConsistentTeacher
from ssod.models.dense_heads.fam3d import FAM3DHead

calls = {"fam3d_forward": 0, "dynamic_assign": 0, "gmm_policy": 0}
fam3d_forward = FAM3DHead.forward
dynamic_assign = DynamicSoftLabelAssigner.assign
gmm_policy = ConsistentTeacher.gmm_policy


def traced_fam3d_forward(self, *args, **kwargs):
    calls["fam3d_forward"] += 1
    return fam3d_forward(self, *args, **kwargs)


def traced_dynamic_assign(self, *args, **kwargs):
    calls["dynamic_assign"] += 1
    return dynamic_assign(self, *args, **kwargs)


def traced_gmm_policy(self, *args, **kwargs):
    calls["gmm_policy"] += 1
    return gmm_policy(self, *args, **kwargs)


FAM3DHead.forward = traced_fam3d_forward
DynamicSoftLabelAssigner.assign = traced_dynamic_assign
ConsistentTeacher.gmm_policy = traced_gmm_policy
sys.argv = ["tools/train.py", *sys.argv[1:]]
runpy.run_path("tools/train.py", run_name="__main__")
trace = (
    "CT_ALGORITHM_TRACE "
    f"FAM3D.forward={calls['fam3d_forward']} "
    f"DynamicSoftLabelAssigner.assign={calls['dynamic_assign']} "
    f"GMM.policy={calls['gmm_policy']}"
)
print(trace)
trace_file = os.environ.get("CT_ALGORITHM_TRACE_FILE")
if trace_file:
    with open(trace_file, "w") as output:
        output.write(trace + "\n")
if not all(calls.values()):
    raise SystemExit("FAM-3D or DynamicSoftLabelAssigner was not called")
