# Reproduction deviations and provenance

This file records choices that differ from, or are not established by, the
CVPR 2023 paper. Update it with measured values after each run.

| Item | Paper | Current setup / status | Reason and likely effect |
|---|---|---|---|
| GPU count | 8 GPUs | Server has 1 RTX 2080 Ti (22 GB) | Hardware-adapted run; cannot establish exact paper optimization. |
| Per-GPU / global batch | 5 / 40 images | One GPU, 5 / 5 images | Keeps 1:4 composition, but has one eighth of the paper global batch and noisier gradients. |
| Split provenance | Random 10% train2017; exact original IDs not printed in paper | Generated using author splitter, fold/seed 1; 11,828 labeled and 106,459 unlabeled images | Do not label it the authors' original split without an ID-level match. |
| Teacher/student augmentation | Weak teacher, strong student | Paper configs explicitly wire weak to teacher and strong to student | The audited upstream 10% configs wire these branches in reverse; correcting this follows paper Fig. 2. |
| Batch sampler | 1 labeled + 4 unlabeled per GPU | Paper configs use `by_prob=False` and ratio `[1,4]` | Upstream `by_prob=True` samples a variable composition per batch. |
| Mean-Teacher threshold | 0.4 | Paper config sets initial and classification threshold to 0.4 | Upstream current baseline config uses 0.5; the paper value takes precedence. |
| LR / schedule | SGD 0.01, constant | Eight-GPU configs keep 0.01; one-GPU configs use 0.00125, fixed | Corrected one-GPU Mean-Teacher smoke at 0.01 logged NaN losses and gradient norm by iteration 50. The corrected 0.00125, 400-step smoke passed with finite losses, positive pseudo boxes, separate EMA teacher/student weights, checkpoint save/resume and validation evaluation. This is a measured hardware adaptation and may change convergence. |
| EMA momentum behavior | 0.9995 | Config cap 0.9995; official hook warms effective momentum from `1 - 1/(iteration+1)` | Training logs show 0.98 at iteration 50 and reach the configured cap later. This is the author hook's implementation behavior. |
| FP16 gradient norm | Not specified | Mean-Teacher smoke had an initial non-finite gradient norm. In the formal run, the logged `grad_norm` aggregate was `inf` in windows ending at iterations 50 and 1,900; every other logged window through 5,750 was finite, and every logged loss scalar was finite. The 4,000-step checkpoint had finite model tensors and AMP metadata `scale=32768`, `_growth_tracker=124`; teacher/student storage was separate and 208 parameter tensors differed. CT at initial scale 65,536 had non-finite norms in 3 of 8 logged windows and scaler scale fell to 256. With scale 512, 2 of 8 logged windows were non-finite and the checker observed 4 skipped updates in 400; losses and all checkpoint tensors stayed finite. | PyTorch 1.9 `GradScaler.step` skips `optimizer.step()` only when it detects inf/NaN in unscaled gradient tensors; the logged window aggregate does not expose per-step `found_inf`, so no exact skipped-update count is claimed for the formal Mean-Teacher run. The 4,000-step checkpoint confirms finite model weights and preserves the scaler state for audit. The validated one-GPU CT setting uses initial scale 512 and the smoke checker enforces a 5% skip budget. |
| Precision | Not specified | Paper baseline inherits official `fp16={}` (MMCV Fp16OptimizerHook default); CT paper config inherits `fp16=None`; one-GPU configs use dynamic FP16 and CT starts at scale 512 | Precision is code-sourced where the paper is silent. The lower CT scale is a hardware stability adaptation validated by a 400-step smoke. |
| Official detector wrapper | Paper describes RetinaNet R50-FPN | Official baseline uses `ATSS` wrapper with `ImprovedRetinaHead` and MaxIoU; reproduction config requests RetinaNet with the same official head | Build/forward compatibility must be confirmed. If incompatible, record and use official wrapper transparently. |
| Best checkpoint retention | Paper reports the final 180k model | Paper configs save the best teacher `bbox_mAP` during author-config validation, plus the final checkpoint | Best is retained for inspection; all paper comparisons use the final 180k checkpoint. |
| Evaluation hook on one GPU | Eight-GPU author setup uses `SubModulesDistEvalHook` | One-GPU configs use MMDetection `EvalHook`; `ssod/apis/train.py` constructs this class directly because MMDetection 2.28.1 does not register it in MMCV `HOOKS`. A one-iteration Mean-Teacher run passed hook construction and saved a checkpoint. The detector wrapper's `inference_on='teacher'` selects teacher predictions. | Fixes the official one-GPU config/registry mismatch; metric and interval remain unchanged. |
| Validation interval | Author baseline: 8,000 steps from 20,000; author full method: 4,000 steps (first at 4,000) | One-GPU configs retain their respective author intervals | Evaluation cadence is code-sourced because the paper does not specify it; it does not change optimizer updates. |
| Test CLI eval options | COCO `dataset.evaluate` accepts dataset metrics | `tools/test.py` strips hook-only kwargs, imports the dynamic assigner registry, and retains ConsistentTeacher's wrapper `train_cfg.num_scores` while clearing the child detector train config | Fixes official config/test-tool issues found during smoke evaluation; no model or metric changes. |
| Official post-paper result | Paper reports 40.0 | README later reports 40.2 | 40.2 is not used as the paper target. |
| Official 2x8 recipe | Paper settings above | Not used as paper recipe | README 2x8 config changes batch ratio, unsupervised weight, EMA, and LR. |

## Interrupted preliminary run

An earlier hardware-adapted Mean-Teacher run used the official baseline branch
assignment and probabilistic sampler. It was stopped at approximately
1,900/180,000 iterations when the task was restarted. It had no checkpoint
(checkpoint interval 4,000), is not a valid smoke test or result, and will not
be resumed or included in `RESULTS.md`. Its log remains on the server for
audit only.

## Split record

Verified on the provided server with `tools/dataset/validate_coco10.py`:

- fold / seed: 1
- train2017: 118,287 images / 860,001 annotations
- labeled: 11,828 images / 86,224 annotations
- unlabeled: 106,459 images / 773,777 annotations
- MMDetection effective labeled dataset: 11,740 images; inherited
  `CocoDataset(filter_empty_gt=True)` removes 88 images with no ground-truth
  boxes. The unlabeled child uses `filter_empty_gt=False` and keeps all 106,459.
- labeled JSON SHA256: `483e558651edb7589a02cd30a85e11e252ad20ec2c818d77b9918e7f740f9120`
- unlabeled JSON SHA256: `b17a66b233799b9e05818ab636db8b3cd45a84d217af25c7fa1a41f9e4d6a1d1`
- image-ID overlap: 0
- union equals all train2017 image IDs: yes
- every annotation references an image in its respective split; annotation union equals train2017: yes
- all 118,287 `train2017` image files were present: yes
- provenance: generated with `semi_coco.py`, seed/fold 1 and seed offset 0; not proven to match the authors' private/original sampled IDs

## Environment-specific differences

Validated server environment: Ubuntu 20.04, Python 3.8.10, PyTorch
1.9.0+cu111, torchvision 0.10.0+cu111, MMCV Full 1.7.1, MMDetection 2.28.1,
NumPy 1.22.3, OpenCV 4.5.5.64, scikit-learn 1.0.2, W&B 0.10.31, and
NVIDIA RTX 2080 Ti (22 GB) with driver 535.216.03. The driver reports CUDA
12.2; CUDA toolkit 11.1.105 and `nvcc` are available at `/usr/local/cuda/bin`.
The full
`pip freeze` snapshot is in `environment/reproduction-lock.txt`. Its server-
local PyTorch wheel URL has been normalized to the pinned public version; use
`environment/install_paper_env.sh` to install the matching CUDA 11.1 wheel.
MMDetection 2.28.1 is checked out from its official v2.28.1 tag because it is
not installed as a pip distribution in this environment. YAPF is pinned to
0.32.0 because 0.40.2 breaks MMCV 1.7.1 config dumping.
