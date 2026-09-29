"""Paper-parameter Mean-Teacher reproduction for COCO-PARTIAL 10%.

Launch with 8 GPUs to match the CVPR 2023 paper's 8 x 5 image global batch.
Implementation-specific detector/head details are inherited from the author's
official baseline config; see REPRODUCTION_DEVIATIONS.md.
"""

_base_ = ["../baseline/mean_teacher_retinanet_r50_fpn_coco_180k_10p.py"]
from configs.reproduction.augmentations import make_unsup_pipeline

# The paper specifies RetinaNet R50-FPN. The official baseline config uses its
# ImprovedRetinaHead with the ATSS detector wrapper and a MaxIoU assigner;
# use RetinaNet as the detector wrapper while keeping the author's head.
model = dict(type="RetinaNet")

# Paper: weak views for the EMA teacher and strong views for the student.
unsup_pipeline = make_unsup_pipeline()

data = dict(
    samples_per_gpu=5,
    train=dict(
        sup=dict(ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json"),
        unsup=dict(
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            pipeline=unsup_pipeline,
        ),
    ),
    sampler=dict(
        train=dict(sample_ratio=[1, 4], by_prob=False),
    ),
)

# The paper explicitly gives a fixed confidence threshold of 0.4 for the
# Mean-Teacher baseline, an unsupervised loss weight of 2, and EMA momentum
# 0.9995. Both threshold fields are set for compatibility with this codebase.
semi_wrapper = dict(
    train_cfg=dict(
        pseudo_label_initial_score_thr=0.4,
        cls_pseudo_threshold=0.4,
        unsup_weight=2.0,
    ),
)
custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.9995, interval=1, warm_up=0),
]

# Paper settings: SGD, lr 0.01, momentum 0.9, weight decay 1e-4,
# no learning-rate decay, 180,000 iterations.
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(_delete_=True, policy="fixed")
runner = dict(type="IterBasedRunner", max_iters=180000)
# The paper reports the final 180k model. Keep the best validation checkpoint
# for completeness, but do not substitute it for the final-iteration result.
evaluation = dict(save_best="bbox_mAP", rule="greater")
# The paper does not specify precision. Retain the author's baseline config's
# `fp16={}` behavior (MMCV Fp16OptimizerHook defaults) as code-sourced detail.
fp16 = dict()

# Keep only the newest two periodic checkpoints. The published baseline
# config retains 20, which would exhaust the server's 50 GB temporary volume
# before this run and Consistent-Teacher can both finish. Best-checkpoint
# retention is managed by EvalHook separately.
checkpoint_config = dict(interval=4000, max_keep_ckpts=2)

# Avoid requiring external W&B credentials for a reproducible local run.
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook", by_epoch=False)])
