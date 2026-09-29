"""Trace GradScaler optimizer decisions and scaler restoration on resume."""

import atexit
import os
from pathlib import Path


def _append_trace(path, line):
    print(line, flush=True)
    if path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a") as stream:
            stream.write(line + "\n")


def _state_value(state, key):
    value = state.get(key) if isinstance(state, dict) else None
    return "none" if value is None else str(value)


def install_amp_optimizer_trace():
    """Observe AMP state without changing model, optimizer, or scaler updates."""
    from torch.cuda.amp import GradScaler
    from mmcv.runner.hooks.optimizer import Fp16OptimizerHook

    trace_path = os.environ.get("CT_AMP_TRACE_FILE")

    current_step = GradScaler.step
    if not getattr(current_step, "_ct_amp_trace", False):
        counts = {"scaler_steps": 0, "applied": 0, "skipped": 0, "unknown": 0}

        def traced_step(self, optimizer, *args, **kwargs):
            # MMCV's Fp16OptimizerHook calls unscale_ before step(). In
            # PyTorch 1.9, this per-optimizer found_inf state controls whether
            # optimizer.step() is actually invoked.
            outcome = None
            per_optimizer = getattr(self, "_per_optimizer_states", None)
            state = per_optimizer.get(id(optimizer)) if per_optimizer is not None else None
            found_inf = state.get("found_inf_per_device") if state is not None else None
            if isinstance(found_inf, dict) and found_inf:
                outcome = any(
                    float(value.detach().item()) != 0.0 for value in found_inf.values()
                )

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

        def write_counts():
            line = (
                "AMP_OPTIMIZER_TRACE "
                f"scaler_steps={counts['scaler_steps']} "
                f"applied={counts['applied']} "
                f"skipped={counts['skipped']} "
                f"unknown={counts['unknown']}"
            )
            _append_trace(trace_path, line)

        traced_step._ct_amp_exit_callback = write_counts
        atexit.register(write_counts)

    current_before_run = Fp16OptimizerHook.before_run
    if not getattr(current_before_run, "_ct_amp_resume_trace", False):

        def traced_before_run(self, runner):
            # Capture the checkpoint metadata before MMCV restores the scaler,
            # then record the actual state immediately after its load_state_dict.
            meta = getattr(runner, "meta", None)
            fp16_meta = meta.get("fp16", {}) if isinstance(meta, dict) else {}
            checkpoint_state = (
                fp16_meta.get("loss_scaler", {}) if isinstance(fp16_meta, dict) else {}
            )
            result = current_before_run(self, runner)
            loaded_state = self.loss_scaler.state_dict()
            phase = os.environ.get("CT_AMP_TRACE_PHASE", "training")
            phase = "".join(c if c.isalnum() or c in "_-" else "_" for c in phase)
            line = (
                "AMP_SCALER_INIT "
                f"phase={phase} "
                f"scale={_state_value(loaded_state, 'scale')} "
                f"growth_tracker={_state_value(loaded_state, '_growth_tracker')} "
                f"meta_scale={_state_value(checkpoint_state, 'scale')} "
                f"meta_growth_tracker={_state_value(checkpoint_state, '_growth_tracker')}"
            )
            _append_trace(trace_path, line)
            return result

        traced_before_run._ct_amp_resume_trace = True
        Fp16OptimizerHook.before_run = traced_before_run
