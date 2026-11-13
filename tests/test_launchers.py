# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module",
    [
        "scripts.causal_lm.lora_fsdp2",
        "scripts.causal_lm.distill_fsdp2",
        "scripts.causal_lm.verify_dataloader",
        "scripts.decision_model.finetune_fsdp2",
        "scripts.decision_model.convert_chat",
        "scripts.decision_model.split_data",
        "scripts.decision_model.evaluate",
        "scripts.decision_model.score",
        "scripts.decision_model.validate_checkpoint",
    ],
)
def test_module_cli_help(module):
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
