#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"
export PYTHONPATH="$workspace_root/mmdetection:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
data_root="${CT_DATA_ROOT:-$workspace_root/data}"
if [[ ! -e "$repo_root/data" ]]; then ln -s "$data_root" "$repo_root/data"; fi

echo "=== GPU ==="
nvidia-smi
echo "=== Python / CUDA toolkit ==="
python --version
command -v nvcc >/dev/null 2>&1 && nvcc --version || echo "nvcc unavailable"
echo "=== Conda ==="
command -v conda >/dev/null 2>&1 && conda env list || echo "conda unavailable"
echo "=== Disk ==="
df -h "$workspace_root"

cd "$repo_root"
python - <<'PY'
import cv2
import mmcv
import mmdet
import numpy as np
import sklearn
import torch
import torchvision
import ssod
try:
    import wandb
    wandb_version = wandb.__version__
except ImportError:
    wandb_version = "not installed"
from mmcv.ops import nms

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("numpy:", np.__version__)
print("opencv:", cv2.__version__)
print("scikit-learn:", sklearn.__version__)
print("wandb:", wandb_version)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the reproduction")
boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], device="cuda", dtype=torch.float32)
scores = torch.tensor([0.9, 0.8], device="cuda")
print("mmcv.ops.nms:", nms(boxes, scores, 0.5)[0].tolist())
PY

python - <<'PY'
from mmcv import Config
from mmdet.models import build_detector
from ssod.datasets import build_dataset
from ssod.utils import patch_config

for name in (
    "mean_teacher_r50_fpn_coco_180k_10p_1gpu.py",
    "consistent_teacher_r50_fpn_coco_180k_10p_1gpu.py",
):
    cfg = Config.fromfile("configs/reproduction/" + name)
    cfg.work_dir = "./work_dirs/environment_check"
    cfg = patch_config(cfg)
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    dataset = build_dataset(cfg.data.train)
    print(name, "built model:", type(model).__name__)
    print("built semi-supervised dataset:", type(dataset).__name__, "length:", len(dataset))
PY

echo "=== pip freeze ==="
lock_path="${CT_ENV_LOCK_OUT:-$repo_root/environment/reproduction-lock.txt}"
mkdir -p "$(dirname "$lock_path")"
{
  printf '%s\n' '# Captured with python -m pip freeze; source checkouts may not appear as distributions.'
  python -m pip freeze | sed -E 's#^torch @ file:.*$#torch==1.9.0+cu111#'
} | tee "$lock_path"
