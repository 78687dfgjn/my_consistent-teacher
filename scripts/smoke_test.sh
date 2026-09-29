#!/usr/bin/env bash
set -euo pipefail

method="${1:-mean_teacher}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"
work_root="${CT_RUN_ROOT:-/hy-tmp/consistent-teacher/runs}/smoke_${method}_$(date +%Y%m%d_%H%M%S)"
data_root="${CT_DATA_ROOT:-$workspace_root/data}"
export PYTHONPATH="$workspace_root/mmdetection:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if [[ ! -e "$repo_root/data" ]]; then ln -s "$data_root" "$repo_root/data"; fi

case "$method" in
  mean_teacher)
    config="configs/reproduction/mean_teacher_r50_fpn_coco_180k_10p_1gpu.py"
    ;;
  consistent_teacher)
    config="configs/reproduction/consistent_teacher_r50_fpn_coco_180k_10p_1gpu.py"
    ;;
  *) echo "Usage: $0 [mean_teacher|consistent_teacher]" >&2; exit 2 ;;
esac

mkdir -p "$work_root"
exec > >(tee -a "$work_root/smoke.log") 2>&1
cd "$repo_root"
python tools/dataset/validate_coco10.py --data-root "$data_root/coco" --percent 10 --fold 1 --check-files
python scripts/verify_batch_ratio.py "$config"
smoke_options=(runner.max_iters=400 checkpoint_config.interval=400 checkpoint_config.max_keep_ckpts=1)
resume_options=(runner.max_iters=401 checkpoint_config.interval=401 checkpoint_config.max_keep_ckpts=1)
if [[ "$method" == consistent_teacher ]]; then
  # The production config warms GMM for 10k steps. Disable that warmup only in
  # the short smoke so the dynamic threshold path is exercised immediately.
  smoke_options+=(semi_wrapper.train_cfg.warmup_step=0)
  resume_options+=(semi_wrapper.train_cfg.warmup_step=0)
fi

# A real 400-step training pass exercises the model, data loader, forward and
# backward paths, EMA and checkpoint hook. Set CT_SMOKE_CHECKPOINT to resume
# the validation part from a completed pass without spending another 400 steps.
if [[ -n "${CT_SMOKE_CHECKPOINT:-}" ]]; then
  checkpoint="$CT_SMOKE_CHECKPOINT"
  training_log_dir="${CT_SMOKE_TRAIN_LOG_DIR:-$(dirname "$checkpoint")}"
else
  if [[ "$method" == consistent_teacher ]]; then
    export CT_ALGORITHM_TRACE_FILE="$work_root/train/algorithm_trace.txt"
    python scripts/train_with_algorithm_trace.py "$config" \
      --work-dir "$work_root/train" --seed 1 --no-validate \
      --cfg-options "${smoke_options[@]}"
  else
    python tools/train.py "$config" \
      --work-dir "$work_root/train" --seed 1 --no-validate \
      --cfg-options "${smoke_options[@]}"
  fi
  checkpoint="$work_root/train/latest.pth"
  training_log_dir="$work_root/train"
fi
test -f "$checkpoint"

python - "$training_log_dir" "$method" <<'PY'
import re
import sys
import math
from pathlib import Path

log_dir = Path(sys.argv[1])
logs = list(log_dir.glob("*.log"))
if not logs:
    raise SystemExit("training log not found")
text = logs[0].read_text(errors="replace")
if re.search(r"(?:sup_|unsup_)?loss(?:_[A-Za-z_]+)?:\s*(?:nan|[-+]?inf)\b", text, re.I):
    raise SystemExit("non-finite training loss found")
norms = re.findall(
    r"Iter \[(\d+)/\d+\].*?grad_norm:\s*(nan|[-+]?inf|[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)",
    text,
    re.I,
)
if not norms:
    raise SystemExit("gradient norm was not logged")
bad_norm_steps = [int(step) for step, value in norms if not math.isfinite(float(value))]
if bad_norm_steps and (bad_norm_steps != [int(norms[0][0])] or len(norms) < 2):
    raise SystemExit(f"gradient norm remained non-finite after initial AMP warmup: {bad_norm_steps}")
if bad_norm_steps:
    print(f"initial gradient norm overflow at iteration {bad_norm_steps[0]}; subsequent logged norms are finite")
if "ema_momentum" not in text:
    raise SystemExit("EMA hook did not report a momentum update")
count_key = "unsup_num_gts"
if not re.search(rf"{count_key}:\s*(?:[1-9]\d*|0?\.\d*[1-9]\d*)", text):
    raise SystemExit(f"no positive pseudo-box count was logged ({count_key})")
for loss_key in ("sup_loss_cls", "sup_loss_bbox", "unsup_loss_cls", "unsup_loss_bbox"):
    if not re.search(rf"{loss_key}:\s*[-+]?\d", text):
        raise SystemExit(f"expected supervised/unsupervised loss missing from log: {loss_key}")
if sys.argv[2] == "consistent_teacher":
    trace_file = log_dir / "algorithm_trace.txt"
    trace_text = trace_file.read_text(errors="replace") if trace_file.exists() else ""
    trace = re.search(r"CT_ALGORITHM_TRACE FAM3D.forward=(\d+) DynamicSoftLabelAssigner.assign=(\d+) GMM.policy=(\d+)", trace_text)
    if not trace or any(int(value) <= 0 for value in trace.groups()):
        raise SystemExit("ASA/FAM-3D/GMM runtime call trace missing")
    thresholds = re.findall(r"(?:unsup_)?gmm_thr:\s*([-+]?\d+(?:\.\d+)?)", text)
    if len(set(thresholds)) < 2:
        raise SystemExit(f"GMM threshold did not visibly change: {thresholds}")
    print("GMM threshold values:", sorted(set(thresholds)))
print("smoke log checks passed: finite losses, EMA logged, pseudo boxes logged")
PY

# Resume the just-saved checkpoint for one iteration. MMDetection logs the
# loaded zero-based index as 399/400 after the 400-step checkpoint; the runner
# then writes the 401-iteration checkpoint to prove one resumed update ran.
python tools/train.py "$config" \
  --work-dir "$work_root/resume" --seed 1 --no-validate \
  --resume-from "$checkpoint" \
  --cfg-options "${resume_options[@]}"

resume_log="$(find "$work_root/resume" -maxdepth 1 -name '*.log' -type f | head -n 1)"
grep -Eq 'resumed from epoch: [0-9]+, iter (399|400)' "$resume_log"
test -f "$work_root/resume/iter_401.pth"
grep -q 'Saving checkpoint at 401 iteration' "$resume_log"

python scripts/verify_checkpoint.py "$config" "$checkpoint"

if [[ -n "${CT_SMOKE_PREDICTIONS:-}" ]]; then
  python scripts/eval_saved_predictions.py \
    "$config" "$CT_SMOKE_PREDICTIONS" "$work_root/evaluation"
else
  bash "$repo_root/scripts/eval_coco.sh" "$config" "$checkpoint" "$work_root/evaluation"
fi
echo "smoke test complete: $work_root"
