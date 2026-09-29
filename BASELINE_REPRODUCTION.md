# Mean-Teacher baseline status

Use the audited configurations and smoke/full-training procedures in
[`REPRODUCTION.md`](REPRODUCTION.md):

- Exact paper settings: `configs/reproduction/mean_teacher_r50_fpn_coco_180k_10p_paper.py`
- One-GPU adaptation: `configs/reproduction/mean_teacher_r50_fpn_coco_180k_10p_1gpu.py`
- Start a validated training run: `scripts/run_mean_teacher_coco10.sh`

The preliminary run that was stopped during the task restart reached about
1,900/180,000 iterations. It used the upstream swapped teacher/student
augmentation mapping and probabilistic per-batch sampling, and ended before
its first checkpoint. It is excluded as a smoke test and as a result. The
corrected paper configs must pass `scripts/smoke_test.sh mean_teacher` before
any full run.
