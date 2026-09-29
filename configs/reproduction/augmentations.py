"""Author pipeline definitions with paper-correct teacher/student branch roles."""

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

_color_transforms = [
    dict(type=name)
    for name in (
        "Identity",
        "AutoContrast",
        "RandEqualize",
        "RandSolarize",
        "RandColor",
        "RandContrast",
        "RandBrightness",
        "RandSharpness",
        "RandPosterize",
    )
]

strong_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="ShuffledSequential",
                transforms=[
                    dict(type="OneOf", transforms=_color_transforms),
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandTranslate", x=(-0.1, 0.1)),
                            dict(type="RandTranslate", y=(-0.1, 0.1)),
                            dict(type="RandRotate", angle=(-30, 30)),
                            [
                                dict(type="RandShear", x=(-30, 30)),
                                dict(type="RandShear", y=(-30, 30)),
                            ],
                        ],
                    ),
                ],
            ),
            dict(type="RandErase", n_iterations=(1, 5), size=[0, 0.2], squared=True),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename", "ori_shape", "img_shape", "img_norm_cfg", "pad_shape",
            "scale_factor", "tag", "transform_matrix",
        ),
    ),
]

weak_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename", "ori_shape", "img_shape", "img_norm_cfg", "pad_shape",
            "scale_factor", "tag", "transform_matrix",
        ),
    ),
]


def make_unsup_pipeline():
    return [
        dict(type="LoadImageFromFile"),
        dict(type="PseudoSamples", with_bbox=True),
        dict(
            type="MultiBranch",
            unsup_teacher=weak_pipeline,
            unsup_student=strong_pipeline,
        ),
    ]
