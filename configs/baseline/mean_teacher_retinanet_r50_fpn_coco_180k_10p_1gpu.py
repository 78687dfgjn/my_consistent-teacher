"""COCO 10% Mean-Teacher baseline adapted for one 22 GB GPU.

The original eight-GPU configuration remains the reference for published AP.
This configuration keeps the model, data split, schedule and EMA settings,
while using a smaller physical batch and a proportionally scaled learning rate.
"""

_base_ = ["./mean_teacher_retinanet_r50_fpn_coco_180k_10p.py"]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    sampler=dict(train=dict(sample_ratio=[1, 1])),
)
optimizer = dict(lr=0.0005)
semi_wrapper = dict(train_cfg=dict(
    pseudo_label_initial_score_thr=0.4,
    cls_pseudo_threshold=0.4,
))
log_config = dict(
    interval=50,
    hooks=[dict(type="TextLoggerHook", by_epoch=False)],
)
fp16 = dict(loss_scale="dynamic")
