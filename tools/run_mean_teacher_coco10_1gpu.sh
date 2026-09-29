#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"
data_root="${CT_DATA_ROOT:-$workspace_root/data}"
work_dir="${CT_WORK_DIR:-$workspace_root/runs/mean_teacher_coco10_1gpu}"

export PYTHONPATH="$workspace_root/mmdetection:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ ! -d "$data_root/coco/train2017" || ! -d "$data_root/coco/val2017" ]]; then
  echo "COCO train2017 and val2017 must be extracted under $data_root/coco" >&2
  exit 1
fi
if [[ ! -f "$data_root/coco/annotations/instances_val2017.json" ]]; then
  echo "COCO validation annotations are missing" >&2
  exit 1
fi

mkdir -p "$data_root/coco/annotations/semi_supervised" "$work_dir"
if [[ ! -f "$data_root/coco/annotations/semi_supervised/instances_train2017.1@10.json" ]]; then
  cd "$repo_root"
  python tools/dataset/semi_coco.py --percent 10 --seed 1 \
    --data-dir "$data_root/coco" \
    --save-dir "$data_root/coco/annotations" --partial-only
fi
ln -sfn "$data_root" "$repo_root/data"

cd "$repo_root"
exec python tools/train.py \
  configs/baseline/mean_teacher_retinanet_r50_fpn_coco_180k_10p_1gpu.py \
  --work-dir "$work_dir" --seed 1
