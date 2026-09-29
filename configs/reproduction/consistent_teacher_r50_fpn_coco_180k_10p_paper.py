"""Paper-parameter Consistent-Teacher reproduction for COCO-PARTIAL 10%.

Launch with 8 GPUs for the paper's 8 x 5 image global batch. This keeps the
author's official FAM3DHead, DynamicSoftLabelAssigner, and class-wise GMM.
"""

_base_ = ["../consistent-teacher/consistent_teacher_r50_fpn_coco_180k_10p.py"]
from configs.reproduction.augmentations import make_unsup_pipeline

# Paper Figure 2: EMA teacher gets weak views; student gets strong views.
unsup_pipeline = make_unsup_pipeline()

data = dict(
    samples_per_gpu=5,
    train=dict(
        sup=dict(
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
        ),
        unsup=dict(
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            pipeline=unsup_pipeline,
        ),
    ),
    sampler=dict(train=dict(sample_ratio=[1, 4], by_prob=False)),
)

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(_delete_=True, policy="fixed")
runner = dict(type="IterBasedRunner", max_iters=180000)
# Keep the best teacher validation checkpoint as an artifact; the reported
# paper comparison still uses the final 180k checkpoint.
evaluation = dict(save_best="bbox_mAP", rule="greater")
fp16 = None
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook", by_epoch=False)])
