# Paper reproduction guide

This repository distinguishes the paper's eight-GPU configuration from a
single-GPU adaptation. Report the latter as hardware-adapted results; it cannot
establish an exact numerical reproduction of the paper.

## Provenance and paper settings

- Paper: Wang et al., CVPR 2023, *Consistent-Teacher: Towards Reducing
  Inconsistent Pseudo-targets in Semi-supervised Object Detection*.
- Author repository commit audited: `b49a9ce1c0ae78d13528be0a9a4eff65a627d581`.
- Paper COCO-PARTIAL protocol: randomly select 10% of `train2017` as labeled,
  use the other 90% as unlabeled, and evaluate AP50:95 on `val2017`.
- The paper specifies 8 GPUs x 5 images/GPU, with one labeled and four
  unlabeled images on each GPU; SGD lr 0.01, momentum 0.9, weight decay
  0.0001, constant LR, 180,000 iterations, unlabeled loss weight 2, EMA
  momentum 0.9995, RetinaNet R50-FPN, and ImageNet pretrained initialization.
- The paper gives a fixed Mean-Teacher confidence threshold of 0.4. The
  Consistent-Teacher experiment uses its class-wise GMM thresholding.
- Paper reference AP50:95: Mean-Teacher 35.5, Consistent-Teacher 40.0.
  The author README reports a later tuned Consistent-Teacher result of 40.2;
  it is not substituted for the paper target.

## Environment

The audited server uses Python 3.8, PyTorch 1.9.0+cu111, MMCV 1.7.1, and
MMDetection 2.28.1. See `environment/reproduction-lock.txt` for the recorded
package set and `scripts/check_environment.sh` for the live checks. The author
README recommends PyTorch 1.9.0 with either MMDetection 2.25.0/MMCV 1.3.9 or
MMDetection 2.28.1/MMCV 1.7.1; this reproduction uses the latter pair. Do not
install MMDetection 3.x for these MMDetection 2.x configs.

For a clean Linux machine, create a Python 3.8 environment and install the
matching CUDA 11.1 PyTorch wheels, then install `mmcv-full==1.7.1` from the
official MMCV wheel index for `cu111/torch1.9.0`, check out MMDetection v2.28.1,
and install both source trees in editable mode. Pin `yapf==0.32.0`; YAPF 0.40.2
is incompatible with MMCV 1.7.1 config dumping. The lock file records the
validated server environment rather than claiming cross-platform portability.

From an environment manager and the directory where you keep source trees:

```bash
conda create -n consistent-teacher python=3.8 -y
conda activate consistent-teacher
git clone git@github.com:78687dfgjn/my_consistent-teacher.git repo
cd repo
environment/install_paper_env.sh
```

The installer places MMDetection v2.28.1 beside the repository if it is not
already checked out there.

## Data and split

Place COCO on the large local disk, outside Git:

```text
/hy-tmp/consistent-teacher/data/coco/
  train2017/
  val2017/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
```

Prepare and validate fold/seed 1:

```bash
export CT_COCO_ROOT=/hy-tmp/consistent-teacher/data/coco
scripts/prepare_coco10.sh
```

The splitter is the author's `tools/dataset/semi_coco.py` with seed 1 and
10 percent. The validation script checks image-ID uniqueness, disjointness,
union against all train2017 image IDs, annotation membership and union, and
image-file existence. The verified files contain 11,828 labeled IDs and
106,459 unlabeled IDs. The official MMDetection `CocoDataset` then filters 88
labeled images with no ground-truth boxes because the inherited labeled config
uses `filter_empty_gt=True`; its effective labeled dataset length is 11,740.
The unlabeled child keeps all 106,459 IDs (`filter_empty_gt=False`). This is
loader behavior after split validation, not a change to the split. Until the
JSON files are proven identical to an
author-provided release, describe this as the generated fold-1/seed-1 split,
not the authors' original split. The split hashes and counts are recorded in
`REPRODUCTION_DEVIATIONS.md` after validation.

On a clean machine, download the official COCO archives directly into the
server data disk and extract them in this layout:

```bash
mkdir -p /hy-tmp/consistent-teacher/data/coco
cd /hy-tmp/consistent-teacher/data/coco
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip
```

Keep the archives and extracted images on the large disk, outside the Git
checkout. Confirm available disk space before extraction and training.

## Config audit

Use the paper configs in `configs/reproduction/` for all reported runs:

- `mean_teacher_r50_fpn_coco_180k_10p_paper.py` is the eight-GPU paper setup.
- `consistent_teacher_r50_fpn_coco_180k_10p_paper.py` is the corresponding
  full method setup with ASA, FAM-3D, and GMM.
- The `*_1gpu.py` configs are separate hardware adaptations for one 22 GB GPU
  and use LR 0.00125 after LR 0.01 produced NaN losses/gradients by the first
  50-step log point; the exact eight-GPU configs retain LR 0.01.

The configs explicitly correct the official configs' swapped weak/strong
branch assignment to match the paper: teacher receives weak augmentation and
student receives strong augmentation. They also enforce exactly 1 labeled + 4
unlabeled images per local batch (`by_prob=False`) and use the common
`data/coco/annotations/semi_supervised` split path. The official Mean-Teacher
config's 0.5 threshold is overridden by the paper's 0.4.

The full method keeps the official `FAM3DHead`, `DynamicSoftLabelAssigner`,
and class-wise GMM score queue. In the author code, the assigner combines focal
classification cost, negative log IoU cost, and center-distance cost, and uses
dynamic-k matching; this is the official implementation of the paper's
classification/regression/distance matching objective. FAM3DHead predicts
spatial deformable offsets and cross-FPN-level interpolation weights. The GMM
uses a 100-score buffer per class and reports `gmm_thr` during training.

The paper-first mapping is:

| Setting | Paper | Author README/config/source | Reproduction decision |
|---|---|---|---|
| Mean-Teacher score threshold | 0.4 (paper Sec. 4, p. 6) | Current baseline config sets both threshold fields to 0.5; `ssod/models/mean_teacher.py` consumes the initial score threshold for teacher boxes | Override both threshold fields to 0.4 in the paper config. |
| Global batch and composition | 8 x 5, each GPU 1 labeled + 4 unlabeled (Sec. 4, p. 5) | Current configs say 5 images/GPU and `[1,4]`, but `by_prob=True` samples a variable count per batch | Set `by_prob=False`; this yields exactly 1+4 locally. |
| Augmentation roles | Weak view for teacher, strong view for student (Fig. 2, p. 4) | Both official 10% configs pass `strong_pipeline` as `unsup_teacher` and `weak_pipeline` as `unsup_student`; model code groups branches by these tags | Correct the branch assignment in the reproduction configs. |
| Optimizer/schedule | SGD 0.01, momentum 0.9, weight decay 1e-4, constant LR, 180k iterations (Sec. 4, p. 5) | Official config has SGD values but encodes `step=[180000,180000]` | Use explicit `policy='fixed'` in the paper configs. |
| EMA | 0.9995 (Sec. 4, p. 5) | `MeanTeacher` hook updates the separate teacher model and its config uses 0.9995 | Keep 0.9995; the smoke checks the hook log and distinct teacher/student state. |
| RetinaNet baseline | RetinaNet R50-FPN (Sec. 4, p. 5) | Author baseline uses `ATSS` wrapper, `ImprovedRetinaHead`, and `MaxIoUAssigner` in the file named `mean_teacher_retinanet...` | Paper config requests the RetinaNet wrapper with the official head; build validation must pass before use. |
| ASA | Eq. (3)/(4): classification + regression + center-distance cost; unified labeled/unlabeled assignment | `DynamicSoftLabelAssigner` uses focal classification, negative log IoU as the regression-quality proxy, center prior, and dynamic-k matching; `FAM3DHead.loss` calls it for both labeled and pseudo boxes | Keep the official implementation and disclose the cost formulation difference. |
| FAM-3D | Spatial plus FPN-scale offset (Eq. (5)/(6), pp. 4-5) | `FAM3DHead` applies deformable spatial sampling and neighboring FPN-scale interpolation | Retain the official head; smoke counts live `FAM3DHead.forward` calls. |
| GMM | Class-specific score mixture and queue around 100 values/class (Sec. 3.4, p. 5) | `ConsistentTeacher` holds a `[num_classes, 100]` score buffer and calls `gmm_policy` | Retain it; smoke counts live GMM calls and requires thresholds to vary. |
| Published number | Paper full model 40.0; Mean-Teacher 35.5; ASA 38.5; FAM-3D 39.5 (Tables 1, 4, 5) | README reports post-submission full-model 40.2 and says hyperparameters were tuned after submission | Compare paper runs to paper values; list README 40.2 only as later context. |

The implementation audit also covers `tools/dataset/semi_coco.py`,
`tools/train.py`, and `tools/test.py`. The generated split is validated by
`tools/dataset/validate_coco10.py`; training code resumes from `latest.pth`
and evaluation runs on the teacher model.

## Validate before full training

```bash
scripts/check_environment.sh
scripts/smoke_test.sh mean_teacher
scripts/smoke_test.sh consistent_teacher
```

Each smoke run performs 400 optimization iterations and checks finite logged
losses, recovered finite gradients, finite checkpoint tensors, EMA logging,
positive pseudo-box counts, checkpoint writing, resume for one iteration,
and a `val2017` bbox evaluation. For the hardware-adapted dynamic FP16 runs,
the checker permits at most 5% scaler-detected skipped updates and records the
count; the eight-GPU paper configs are not changed. The
Consistent-Teacher smoke also counts live calls to FAM3DHead, the dynamic
assigner, and GMM policy, and requires at least two distinct `gmm_thr` values
before its full run.

## Full training

Run long jobs in `tmux` (or another persistent terminal). Scripts append stdout
and stderr to `/hy-tmp/consistent-teacher/logs/`, write a PID file, and resume
from `latest.pth` when present. Check available disk space before starting.
Validation follows each author config: Mean-Teacher evaluates every 8,000
iterations starting at 20,000; Consistent-Teacher evaluates every 4,000
iterations (first evaluation at 4,000). Both save the best teacher by
`bbox_mAP`. Exact eight-GPU configs use the author's distributed submodule
hook; one-GPU configs use MMDetection's single-GPU `EvalHook`, with the
wrapper's `inference_on='teacher'` setting. The test utility strips
training-hook-only evaluation settings before calling the COCO dataset
evaluator, including the author's `evaluated_modules` field.
The reported paper comparison
must use the final iteration-180,000 checkpoint; the best checkpoint is kept
as a separate artifact and is not selected to improve the reported number.

Exact paper setup, only on eight GPUs:

```bash
tmux new -d -s mean_teacher_coco10 \
  'cd /hy-tmp/consistent-teacher/repo && scripts/run_mean_teacher_coco10.sh paper'
tmux new -d -s consistent_teacher_coco10 \
  'cd /hy-tmp/consistent-teacher/repo && scripts/run_consistent_teacher_coco10.sh paper'
```

One-GPU adapted setup:

```bash
tmux new -d -s mean_teacher_coco10 \
  'cd /hy-tmp/consistent-teacher/repo && scripts/run_mean_teacher_coco10.sh 1gpu'
tmux new -d -s consistent_teacher_coco10 \
  'cd /hy-tmp/consistent-teacher/repo && scripts/run_consistent_teacher_coco10.sh 1gpu'
```

The single-GPU setup must only be reported with its actual GPU count, batch,
precision, learning rate, iterations, and deviations. Do not compare it as an
exact eight-GPU reproduction.

On the verified one-RTX-2080-Ti host, the Mean-Teacher run is currently
scheduled first, followed by evaluation and the full Consistent-Teacher run.
The observed early Mean-Teacher throughput was about 0.71 seconds per
iteration, so the 180,000-step run takes roughly 37 hours before validation
overhead. Both jobs write periodic checkpoints and append logs under
`/hy-tmp/consistent-teacher/`.

To run the two adapted experiments sequentially, evaluate each final
checkpoint, and retry a failed training stage from `latest.pth`, start the
sequence supervisor:

```bash
tmux new -d -s reproduction_sequence \
  'cd /hy-tmp/consistent-teacher/repo && bash scripts/run_reproduction_sequence_1gpu.sh'
```

The supervisor detects an already-running Mean-Teacher session, then waits for
its 180,000-step checkpoint before evaluation. It starts Consistent-Teacher
only when the final Mean-Teacher AP is within one point of the paper's 35.5;
otherwise it pauses after logging the measured result so the baseline can be
diagnosed first. Its progress is logged to
`/hy-tmp/consistent-teacher/logs/reproduction_sequence_1gpu.log`.

## Evaluation

Evaluate the final teacher model using the matching reproduction config:

```bash
scripts/eval_coco.sh \
  configs/reproduction/mean_teacher_r50_fpn_coco_180k_10p_1gpu.py \
  /hy-tmp/consistent-teacher/runs/mean_teacher_coco10_1gpu/iter_180000.pth
scripts/eval_coco.sh \
  configs/reproduction/consistent_teacher_r50_fpn_coco_180k_10p_1gpu.py \
  /hy-tmp/consistent-teacher/runs/consistent_teacher_coco10_1gpu/iter_180000.pth
```

Both configs set inference to the teacher. Keep the raw evaluation log,
prediction pickle, checkpoint, and metric output under `/hy-tmp`; do not add
them to Git. Enter the verified AP50:95 values and exact checkpoint paths in
`RESULTS.md`.
