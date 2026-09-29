"""Unit test the GradScaler found_inf decision counter without CUDA."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "amp_optimizer_trace.py"


class FakeValue:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def item(self):
        return self.value


class FakeOptimizer:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class FakeGradScaler:
    def __init__(self):
        self._per_optimizer_states = {}

    def step(self, optimizer):
        state = self._per_optimizer_states[id(optimizer)]
        found_inf = state["found_inf_per_device"]
        if not any(value.item() for value in found_inf.values()):
            optimizer.step()


class AmpOptimizerTraceTest(unittest.TestCase):
    def test_counts_applied_and_skipped_steps_from_found_inf(self):
        fake_torch = types.ModuleType("torch")
        fake_cuda = types.ModuleType("torch.cuda")
        fake_amp = types.ModuleType("torch.cuda.amp")
        fake_amp.GradScaler = FakeGradScaler
        fake_cuda.amp = fake_amp
        fake_torch.cuda = fake_cuda

        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "torch.cuda": fake_cuda, "torch.cuda.amp": fake_amp},
        ):
            spec = importlib.util.spec_from_file_location("amp_optimizer_trace_under_test", SCRIPT)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.install_amp_optimizer_trace()

            scaler = FakeGradScaler()
            optimizer = FakeOptimizer()
            scaler._per_optimizer_states[id(optimizer)] = {
                "found_inf_per_device": {"cuda:0": FakeValue(0.0)}
            }
            scaler.step(optimizer)
            scaler._per_optimizer_states[id(optimizer)] = {
                "found_inf_per_device": {"cuda:0": FakeValue(1.0)}
            }
            scaler.step(optimizer)

            self.assertEqual(optimizer.steps, 1)
            self.assertEqual(
                FakeGradScaler.step._ct_amp_counts,
                {"scaler_steps": 2, "applied": 1, "skipped": 1, "unknown": 0},
            )


if __name__ == "__main__":
    unittest.main()
