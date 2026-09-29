#!/usr/bin/env python3
"""Run the standard trainer while recording GradScaler update decisions."""

import runpy
import sys

from amp_optimizer_trace import install_amp_optimizer_trace


def main():
    install_amp_optimizer_trace()
    sys.argv = ["tools/train.py", *sys.argv[1:]]
    runpy.run_path("tools/train.py", run_name="__main__")


if __name__ == "__main__":
    main()
