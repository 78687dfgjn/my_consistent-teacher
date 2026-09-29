# Mean-Teacher COCO 10% baseline reproduction

This reproduces the paper's Mean-Teacher baseline using the upstream
`configs/baseline/mean_teacher_retinanet_r50_fpn_coco_180k_10p.py` model.
The paper reports 35.5 COCO val2017 AP for the eight-GPU reference run.
The included single-GPU configuration is an adaptation for a 22 GB GPU:
two images per step (one labeled, one unlabeled), learning rate 0.0005,
FP16 training, and the paper's fixed pseudo-label threshold of 0.4.
The changed global batch means its final AP is not directly comparable
with the paper's eight-GPU result.

## Dataset and storage

Keep the large COCO archives, extracted images, annotations, logs, and
checkpoints under `/hy-tmp/consistent-teacher` on the server. The code
checkout and the MMDetection 2.28.1 checkout should be sibling directories:

```text
/hy-tmp/consistent-teacher/
  repo/
  mmdetection/
  data/coco/train2017/
  data/coco/val2017/
  data/coco/annotations/
  runs/
```

The dataset comes from the official COCO 2017 `train2017.zip`,
`val2017.zip`, and `annotations_trainval2017.zip`. Generate the split
with seed 1 and no random offset; `--partial-only` does not require
the separate `unlabeled2017` dataset:

```bash
cd /hy-tmp/consistent-teacher/repo
python tools/dataset/semi_coco.py --percent 10 --seed 1 \
  --data-dir /hy-tmp/consistent-teacher/data/coco \
  --save-dir /hy-tmp/consistent-teacher/data/coco/annotations --partial-only
```

## Environment and training

The tested server has PyTorch 1.9.0+cu111, torchvision 0.10.0+cu111,
MMCV Full 1.7.1, MMDetection 2.28.1, YAPF 0.32.0, Python 3.8, and one
RTX 2080 Ti. Newer YAPF releases remove the `verify` argument used by
this MMCV release.
Because the server's editable installs are not visible on its system
Python path, the launcher adds both source directories to `PYTHONPATH`.

```bash
cd /hy-tmp/consistent-teacher/repo
bash tools/run_mean_teacher_coco10_1gpu.sh
```

The launcher checks the extracted dataset and writes training logs and
checkpoints to `/hy-tmp/consistent-teacher/runs/mean_teacher_coco10_1gpu`.
The original config remains available for an eight-GPU reference run.
