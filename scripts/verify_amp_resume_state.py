#!/usr/bin/env python3
"""Require the AMP scaler state loaded at resume to match its checkpoint."""

import argparse
import math
import re
from pathlib import Path

import torch


INIT_RE = re.compile(
    r"AMP_SCALER_INIT phase=resume "
    r"scale=(\S+) growth_tracker=(\S+) "
    r"meta_scale=(\S+) meta_growth_tracker=(\S+)"
)


def as_number(value, name):
    if value is None or str(value).lower() == "none":
        raise SystemExit(f"missing AMP scaler field: {name}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"invalid AMP scaler field {name}: {value!r}")


def same_number(left, right):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("trace")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    expected = meta.get("fp16", {}).get("loss_scaler", {})
    expected_scale = as_number(expected.get("scale"), "checkpoint scale")
    expected_tracker = as_number(
        expected.get("_growth_tracker"), "checkpoint growth_tracker"
    )

    lines = Path(args.trace).read_text(errors="replace").splitlines()
    matches = [INIT_RE.search(line) for line in lines]
    matches = [match for match in matches if match]
    if len(matches) != 1:
        raise SystemExit(
            f"expected one AMP resume initialization trace, found {len(matches)}"
        )
    loaded_scale, loaded_tracker, meta_scale, meta_tracker = map(float, matches[0].groups())

    observed = {
        "hook-loaded scale": loaded_scale,
        "runner metadata scale": meta_scale,
    }
    if not all(same_number(value, expected_scale) for value in observed.values()):
        raise SystemExit(
            f"AMP scale was not restored from checkpoint: expected {expected_scale}, "
            f"observed {observed}"
        )
    observed_trackers = {
        "hook-loaded growth_tracker": loaded_tracker,
        "runner metadata growth_tracker": meta_tracker,
    }
    if not all(same_number(value, expected_tracker) for value in observed_trackers.values()):
        raise SystemExit(
            f"AMP growth tracker was not restored from checkpoint: "
            f"expected {expected_tracker}, observed {observed_trackers}"
        )
    print(
        "AMP resume state restored before the next optimizer update: "
        f"scale={expected_scale:g}, growth_tracker={expected_tracker:g}"
    )


if __name__ == "__main__":
    main()
