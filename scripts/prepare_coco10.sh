#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
coco_root="${CT_COCO_ROOT:-/hy-tmp/consistent-teacher/data/coco}"
fold="${CT_FOLD:-1}"
percent="${CT_PERCENT:-10}"

if [[ ! -f "$coco_root/annotations/instances_train2017.json" ]]; then
  echo "Missing COCO train annotations under $coco_root/annotations" >&2
  exit 1
fi

mkdir -p "$coco_root/annotations/semi_supervised"
cd "$repo_root"
python tools/dataset/semi_coco.py \
  --percent "$percent" --seed "$fold" --seed-offset 0 \
  --data-dir "$coco_root" --save-dir "$coco_root/annotations" --partial-only
python tools/dataset/validate_coco10.py \
  --data-root "$coco_root" --percent "$percent" --fold "$fold" --check-files
