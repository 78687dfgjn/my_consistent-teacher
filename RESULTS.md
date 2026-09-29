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
| Mean-Teacher hardware smoke | Passed (400 iterations) | Finite losses; first 50-step log window had positive pseudo boxes; EMA momentum hook active; teacher/student checkpoint tensors are separate and differ; 401st-step resume saved; val2017 prediction cache evaluated. A separate fixed sample of 8 unlabeled weak views at step 400 had 0 accepted boxes (mean per-image max score 0.123; range 0.076–0.187 at threshold 0.4). The smoke checkpoint's AP is 0.0001 and is not a reproduction result. |
| Consistent-Teacher hardware smoke (initial scale 65,536) | Algorithm path verified; stability gate failed | FAM3D.forward=1,200, DynamicSoftLabelAssigner.assign=2,000, GMM.policy=484; logged losses were finite and GMM thresholds changed, but 8/400 dynamic-FP16 gradient-norm overflows exceeded the initial gate. Scale reached 256 and all checkpoint tensors were finite. |
| Consistent-Teacher hardware smoke (initial scale 512) | Passed (400 iterations) | FAM3D.forward=1,200, DynamicSoftLabelAssigner.assign=2,000, GMM.policy=408; class thresholds varied from 0.0227 to 0.1294; losses and checkpoint tensors finite; 4/400 AMP updates skipped (1%, within the documented 5% smoke budget); resume to 401 and val2017 evaluation completed. Smoke AP is 0.0000 and is not a reproduction result. |
| Mean-Teacher full run | Running | Started 2026-09-29 13:32 server time; reached 3,250/180,000. Losses finite and gradient norm finite from step 100 onward; GPU utilization about 84%, observed memory 13.7 GB, throughput about 0.73 s/step (roughly 36.5 hours total before validation at the current rate). `unsup_num_gts` has been positive at steps 50 (0.14 average), 2,050 (0.005), 2,200 (0.005), 2,250 (0.005), 2,400 (0.005), 2,450 (0.005), 2,600 (0.01), 2,650 (0.015), 2,700 (0.005), 2,750 (0.015), 2,850 (0.01), 2,900 (0.025), 2,950 (0.035), 3,000 (0.005), 3,050 (0.03), 3,100 (0.035), 3,150 (0.02), 3,200 (0.03) and 3,250 (0.045); corresponding pseudo-box regression losses were 0.04071, 0.00029, 0.00024, 0.00046, 0.00021, 0.00030, 0.00071, 0.00088, 0.00024, 0.00094, 0.00071, 0.00153, 0.00244, 0.00038, 0.00192, 0.02496, 0.02794, 0.03508 and 0.03107. Pseudo boxes were logged in each of the latest three windows, while the average accepted count remains low. Runtime config dump confirms weak `RandResize`/`RandFlip` teacher views and strong color/geometric/erase student views. A tmux watcher will run the fixed-sample diagnostic when `iter_8000.pth` appears. |
| Consistent-Teacher full run | Queued after Mean-Teacher evaluation | Smoke validation passed; full training starts after Mean-Teacher finishes and is evaluated, as requested. |
