"""Unit test AMP update counting and resume-state observation without CUDA."""

import atexit
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
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


def fake_imports():
    class FakeGradScaler:
        def __init__(self):
            self._per_optimizer_states = {}

        def step(self, optimizer):
            state = self._per_optimizer_states[id(optimizer)]
            found_inf = state["found_inf_per_device"]
            if not any(value.item() for value in found_inf.values()):
                optimizer.step()

    class FakeLossScaler:
        def __init__(self):
            self._state = {"scale": 65536.0, "_growth_tracker": 0}

        def state_dict(self):
            return dict(self._state)

        def load_state_dict(self, state):
            self._state = dict(state)

    class FakeFp16OptimizerHook:
        def __init__(self):
            self.loss_scaler = FakeLossScaler()

        def before_run(self, runner):
            state = runner.meta["fp16"]["loss_scaler"]
            self.loss_scaler.load_state_dict(state)

    fake_torch = types.ModuleType("torch")
    fake_cuda = types.ModuleType("torch.cuda")
    fake_amp = types.ModuleType("torch.cuda.amp")
    fake_amp.GradScaler = FakeGradScaler
    fake_cuda.amp = fake_amp
    fake_torch.cuda = fake_cuda

    fake_mmcv = types.ModuleType("mmcv")
    fake_runner = types.ModuleType("mmcv.runner")
    fake_hooks = types.ModuleType("mmcv.runner.hooks")
    fake_optimizer = types.ModuleType("mmcv.runner.hooks.optimizer")
    fake_optimizer.Fp16OptimizerHook = FakeFp16OptimizerHook
    fake_hooks.optimizer = fake_optimizer
    fake_runner.hooks = fake_hooks
    fake_mmcv.runner = fake_runner

    modules = {
        "torch": fake_torch,
        "torch.cuda": fake_cuda,
        "torch.cuda.amp": fake_amp,
        "mmcv": fake_mmcv,
        "mmcv.runner": fake_runner,
        "mmcv.runner.hooks": fake_hooks,
        "mmcv.runner.hooks.optimizer": fake_optimizer,
    }
    return modules, FakeGradScaler, FakeFp16OptimizerHook


class AmpOptimizerTraceTest(unittest.TestCase):
    def load_trace_module(self):
        spec = importlib.util.spec_from_file_location("amp_optimizer_trace_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_counts_applied_and_skipped_steps_from_found_inf(self):
        modules, FakeGradScaler, _ = fake_imports()
        with patch.dict(sys.modules, modules):
            module = self.load_trace_module()
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

    def test_records_scaler_state_loaded_from_resume_metadata(self):
        modules, FakeGradScaler, FakeFp16OptimizerHook = fake_imports()
        with tempfile.TemporaryDirectory() as directory:
            trace_path = str(Path(directory) / "amp.log")
            environment = {
                "CT_AMP_TRACE_FILE": trace_path,
                "CT_AMP_TRACE_PHASE": "resume",
            }
            with patch.dict(sys.modules, modules), patch.dict(os.environ, environment):
                module = self.load_trace_module()
                module.install_amp_optimizer_trace()
                runner = SimpleNamespace(
                    meta={
                        "fp16": {
                            "loss_scaler": {
                                "scale": 16384.0,
                                "_growth_tracker": 1461,
                            }
                        }
                    }
                )
                FakeFp16OptimizerHook().before_run(runner)
                atexit.unregister(FakeGradScaler.step._ct_amp_exit_callback)

            trace = Path(trace_path).read_text()
            self.assertIn(
                "AMP_SCALER_INIT phase=resume scale=16384.0 growth_tracker=1461 "
                "meta_scale=16384.0 meta_growth_tracker=1461",
                trace,
            )


if __name__ == "__main__":
    unittest.main()
