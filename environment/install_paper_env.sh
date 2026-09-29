#!/usr/bin/env bash
set -euo pipefail

# Run from the parent directory of this Consistent-Teacher checkout inside a
# clean Python 3.8 environment with a CUDA 11.1-compatible NVIDIA driver.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"

python -m pip install --upgrade 'pip<25'
python -m pip install \
  torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  --extra-index-url https://download.pytorch.org/whl/cu111
python -m pip install mmcv-full==1.7.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
python -m pip install -r "$repo_root/requirements-reproduction.txt"

if [[ ! -d "$workspace_root/mmdetection" ]]; then
  git clone --branch v2.28.1 --depth 1 \
    https://github.com/open-mmlab/mmdetection.git "$workspace_root/mmdetection"
fi
python -m pip install -e "$workspace_root/mmdetection"
python -m pip install -e "$repo_root" --no-deps

echo "Install finished. Run scripts/check_environment.sh before training."
