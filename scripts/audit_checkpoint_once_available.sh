#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <config.py> <checkpoint.pth> <audit-log>" >&2
  exit 2
fi

config="$1"
checkpoint="$2"
audit_log="$3"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"

while [[ ! -s "$checkpoint" ]]; do
  sleep 30
done
# Let the checkpoint writer finish before loading the file.
sleep 30

mkdir -p "$(dirname "$audit_log")"
export PYTHONPATH="$workspace_root/mmdetection:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
python scripts/verify_checkpoint.py "$config" "$checkpoint" 2>&1 | tee "$audit_log"
