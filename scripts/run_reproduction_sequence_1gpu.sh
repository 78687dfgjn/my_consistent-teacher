#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="${CT_RUN_ROOT:-/hy-tmp/consistent-teacher/runs}"
log_root="${CT_LOG_ROOT:-/hy-tmp/consistent-teacher/logs}"
mkdir -p "$run_root" "$log_root"
exec > >(tee -a "$log_root/reproduction_sequence_1gpu.log") 2>&1

wait_for_final_checkpoint() {
    local session="$1" runner="$2" checkpoint="$3" attempt
    for attempt in 1 2 3; do
        if [[ -f "$checkpoint" ]]; then
            echo "$(date -Is) final checkpoint ready: $checkpoint"
            return 0
        fi
        if ! tmux has-session -t "$session" 2>/dev/null; then
            echo "$(date -Is) launching/resuming $runner (attempt $attempt)"
            tmux new-session -d -s "$session" \
                "cd '$repo_root' && bash '$repo_root/scripts/$runner' 1gpu"
        else
            echo "$(date -Is) monitoring active tmux session $session"
        fi
        while tmux has-session -t "$session" 2>/dev/null; do
            sleep 60
        done
        [[ -f "$checkpoint" ]] && continue
        echo "$(date -Is) $session ended without final checkpoint; see $log_root"
        sleep 30
    done
    echo "$(date -Is) failed to produce $checkpoint after three attempts" >&2
    return 1
}

run_eval() {
    local name="$1" config="$2" checkpoint="$3" attempt
    for attempt in 1 2 3; do
        echo "$(date -Is) evaluating $name (attempt $attempt)"
        if bash "$repo_root/scripts/eval_coco.sh" "$config" "$checkpoint"; then
            echo "$(date -Is) evaluation finished: $name"
            return 0
        fi
        sleep 30
    done
    echo "$(date -Is) evaluation failed three times: $name" >&2
    return 1
}

wait_for_final_checkpoint \
    mean_teacher_full run_mean_teacher_coco10.sh \
    "$run_root/mean_teacher_coco10_1gpu/iter_180000.pth"
run_eval mean_teacher \
    "$repo_root/configs/reproduction/mean_teacher_r50_fpn_coco_180k_10p_1gpu.py" \
    "$run_root/mean_teacher_coco10_1gpu/iter_180000.pth"

wait_for_final_checkpoint \
    consistent_teacher_full run_consistent_teacher_coco10.sh \
    "$run_root/consistent_teacher_coco10_1gpu/iter_180000.pth"
run_eval consistent_teacher \
    "$repo_root/configs/reproduction/consistent_teacher_r50_fpn_coco_180k_10p_1gpu.py" \
    "$run_root/consistent_teacher_coco10_1gpu/iter_180000.pth"

echo "$(date -Is) Mean-Teacher and Consistent-Teacher training/evaluation sequence completed"
