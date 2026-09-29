# Reproduction results

No complete reproduction result is claimed until the 180,000-iteration run
finishes and its final teacher checkpoint is evaluated on COCO `val2017`.

| Experiment | Paper mAP (AP50:95) | Reproduced mAP | Difference | GPU setup | Effective batch | Iterations | Training time | Peak VRAM | Config | Checkpoint | Notes |
|---|---:|---:|---:|---|---:|---:|---|---:|---|---|---|
| Mean-Teacher COCO10 | 35.5 | pending | pending | 1 x RTX 2080 Ti, 22 GB (hardware-adapted) | 5 | 180,000 | pending | pending | `configs/reproduction/mean_teacher_r50_fpn_coco_180k_10p_1gpu.py` | pending | Paper's 8 x 5 batch cannot be matched on this server. |
| Consistent-Teacher COCO10 | 40.0 | pending | pending | 1 x RTX 2080 Ti, 22 GB (hardware-adapted) | 5 | 180,000 | pending | pending | `configs/reproduction/consistent_teacher_r50_fpn_coco_180k_10p_1gpu.py` | pending | ASA/FAM-3D/GMM must be verified active. |

## Run record

| Run | Status | Evidence |
|---|---|---|
| Preliminary Mean-Teacher (invalid) | Interrupted at ~1,900 iterations | Wrong teacher/student augmentation mapping and probabilistic per-batch sampler; no checkpoint. Excluded above. |
| Mean-Teacher hardware smoke | Passed (400 iterations) | Finite losses; positive pseudo boxes logged; EMA momentum hook active; teacher/student checkpoint tensors are separate and differ; 401st-step resume saved; val2017 prediction cache evaluated. The smoke checkpoint's AP is 0.0001 and is not a reproduction result. |
| Consistent-Teacher hardware smoke (initial scale 65,536) | Algorithm path verified; stability gate failed | FAM3D.forward=1,200, DynamicSoftLabelAssigner.assign=2,000, GMM.policy=484; logged losses were finite and GMM thresholds changed, but 8/400 dynamic-FP16 gradient-norm overflows exceeded the initial gate. Scale reached 256 and all checkpoint tensors were finite. Repeating with initial scale 512 before full training. |
| Mean-Teacher full run | Not yet run | Requires passing smoke tests. |
| Consistent-Teacher full run | Not yet run | Requires passing smoke tests including GMM threshold changes. |
