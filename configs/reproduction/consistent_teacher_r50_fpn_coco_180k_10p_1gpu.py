"""Single-GPU hardware adaptation of the paper Consistent-Teacher config.

One 22 GB GPU processes five images per update instead of the paper's global
batch of 40. It uses the hardware-adapted LR because the paper LR diverged in
the corrected single-GPU Mean-Teacher smoke. This is not an exact numerical
reproduction. See REPRODUCTION_DEVIATIONS.md.
"""

_base_ = ["./consistent_teacher_r50_fpn_coco_180k_10p_paper.py"]

data = dict(samples_per_gpu=5, workers_per_gpu=2)
optimizer = dict(lr=0.00125)
fp16 = dict(_delete_=True, loss_scale=dict(init_scale=512.0))
evaluation = dict(
    _delete_=True,
    type="EvalHook",
    interval=4000,
    save_best="bbox_mAP",
    rule="greater",
)
