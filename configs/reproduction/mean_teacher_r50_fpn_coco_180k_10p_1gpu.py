"""Single-GPU hardware adaptation of the paper Mean-Teacher config.

This does not match the paper's global batch of 40: one 22 GB GPU processes
five images per update. The paper LR produced non-finite losses at the first
50-step log point, so this hardware config uses the previously observed stable
single-GPU LR pending corrected smoke validation. See
REPRODUCTION_DEVIATIONS.md.
"""

_base_ = ["./mean_teacher_r50_fpn_coco_180k_10p_paper.py"]

data = dict(samples_per_gpu=5, workers_per_gpu=2)
optimizer = dict(lr=0.00125)
fp16 = dict(loss_scale="dynamic")
evaluation = dict(
    _delete_=True,
    type="EvalHook",
    interval=8000,
    start=20000,
    save_best="bbox_mAP",
    rule="greater",
)
