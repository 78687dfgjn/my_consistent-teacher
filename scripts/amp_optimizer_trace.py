"""Count real optimizer updates skipped by PyTorch's dynamic GradScaler."""

import atexit
import os
from pathlib import Path


def install_amp_optimizer_trace():
    """Trace GradScaler decisions without wrapping or changing optimizer.step."""
    from torch.cuda.amp import GradScaler

    current_step = GradScaler.step
    if getattr(current_step, "_ct_amp_trace", False):
        return

    counts = {"scaler_steps": 0, "applied": 0, "skipped": 0, "unknown": 0}
    trace_path = os.environ.get("CT_AMP_TRACE_FILE")

    def traced_step(self, optimizer, *args, **kwargs):
        # MMCV's Fp16OptimizerHook calls unscale_ before step(). In PyTorch
        # 1.9, GradScaler uses these per-optimizer found_inf tensors to decide
        # whether optimizer.step() is invoked.
        outcome = None
        per_optimizer = getattr(self, "_per_optimizer_states", None)
        state = per_optimizer.get(id(optimizer)) if per_optimizer is not None else None
        found_inf = state.get("found_inf_per_device") if state is not None else None
        if isinstance(found_inf, dict) and found_inf:
            outcome = any(float(value.detach().item()) != 0.0 for value in found_inf.values())

        result = current_step(self, optimizer, *args, **kwargs)
        counts["scaler_steps"] += 1
        if outcome is None:
            counts["unknown"] += 1
        elif outcome:
            counts["skipped"] += 1
        else:
            counts["applied"] += 1
        return result

    traced_step._ct_amp_trace = True
    traced_step._ct_amp_counts = counts
    GradScaler.step = traced_step

    def write_trace():
        line = (
            "AMP_OPTIMIZER_TRACE "
            f"scaler_steps={counts['scaler_steps']} "
            f"applied={counts['applied']} "
            f"skipped={counts['skipped']} "
            f"unknown={counts['unknown']}"
        )
        print(line, flush=True)
        if trace_path:
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(line + "\n")

    atexit.register(write_trace)
