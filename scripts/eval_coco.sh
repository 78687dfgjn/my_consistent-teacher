#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config.py> <checkpoint.pth> [output-dir]" >&2
  exit 2
fi
config="$1"
checkpoint="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"
out_dir="${3:-/hy-tmp/consistent-teacher/runs/evaluation_$(date +%Y%m%d_%H%M%S)}"
export PYTHONPATH="$workspace_root/mmdetection:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$out_dir"
cd "$repo_root"
python tools/test.py "$config" "$checkpoint" \
  --eval bbox --out "$out_dir/predictions.pkl" --work-dir "$out_dir" \
  2>&1 | tee "$out_dir/evaluation.log"
