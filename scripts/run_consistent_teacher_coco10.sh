#!/usr/bin/env bash
set -euo pipefail

mode="${1:-paper}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"
data_root="${CT_DATA_ROOT:-$workspace_root/data}"
run_root="${CT_RUN_ROOT:-/hy-tmp/consistent-teacher/runs}"
log_root="${CT_LOG_ROOT:-/hy-tmp/consistent-teacher/logs}"
export PYTHONPATH="$workspace_root/mmdetection:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ ! -d "$data_root/coco/train2017" || ! -d "$data_root/coco/val2017" ]]; then
  echo "COCO images must be extracted under $data_root/coco" >&2
  exit 1
fi
if [[ ! -f "$data_root/coco/annotations/semi_supervised/instances_train2017.1@10.json" ]]; then
  CT_COCO_ROOT="$data_root/coco" "$repo_root/scripts/prepare_coco10.sh"
fi
python "$repo_root/tools/dataset/validate_coco10.py" --data-root "$data_root/coco" --percent 10 --fold 1 --check-files
if [[ ! -e "$repo_root/data" ]]; then ln -s "$data_root" "$repo_root/data"; fi

case "$mode" in
  paper)
    config="configs/reproduction/consistent_teacher_r50_fpn_coco_180k_10p_paper.py"
    gpus="${NUM_GPUS:-8}"
    if [[ "$gpus" != 8 ]]; then echo "Paper config requires 8 GPUs (requested $gpus)" >&2; exit 2; fi
    available_gpus="$(nvidia-smi -L | wc -l)"
    if [[ "$available_gpus" -lt 8 ]]; then echo "Paper config needs 8 visible GPUs; found $available_gpus" >&2; exit 2; fi
    run_dir="${CT_RUN_DIR:-$run_root/consistent_teacher_coco10_paper_8gpu}"
    ;;
  1gpu)
    config="configs/reproduction/consistent_teacher_r50_fpn_coco_180k_10p_1gpu.py"
    gpus=1
    run_dir="${CT_RUN_DIR:-$run_root/consistent_teacher_coco10_1gpu}"
    ;;
  *) echo "Usage: $0 [paper|1gpu]" >&2; exit 2 ;;
esac

mkdir -p "$run_dir" "$log_root"
log_file="$log_root/$(basename "$run_dir").log"
exec > >(tee -a "$log_file") 2>&1
echo "config=$config run_dir=$run_dir mode=$mode started=$(date -Is)"
nvidia-smi
df -h "$run_root"
printf '%s\n' "$$" > "$log_file.pid"

resume_args=()
if [[ -f "$run_dir/latest.pth" ]]; then
  resume_args=(--resume-from "$run_dir/latest.pth")
elif [[ -n "${CT_RESUME_FROM:-}" ]]; then
  if [[ ! -f "$CT_RESUME_FROM" ]]; then echo "Resume checkpoint not found: $CT_RESUME_FROM" >&2; exit 2; fi
  resume_args=(--resume-from "$CT_RESUME_FROM")
fi
cd "$repo_root"
if [[ "$mode" == paper ]]; then
  bash tools/dist_train.sh "$config" "$gpus" --work-dir "$run_dir" --seed 1 "${resume_args[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python tools/train.py "$config" --work-dir "$run_dir" --seed 1 "${resume_args[@]}"
fi
