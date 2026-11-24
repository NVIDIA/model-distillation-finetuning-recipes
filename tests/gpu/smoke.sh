#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-GPU smoke test, run by .github/workflows/gpu.yml inside the docker/Dockerfile
# image from the repository root. Uses small public Qwen3 models pinned to fixed
# revisions and a generated dataset. Any failed step fails the script.
set -euo pipefail

OUT=${OUT:-/tmp/mdfr-smoke}
export HF_HOME=${HF_HOME:-$OUT/hf} WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
step() { echo; echo "=== $*"; }

step "CPU suite"
python -m pip install --quiet "pytest>=7"
python -m pytest -q -p no:cacheprovider -m "not gpu and not distributed and not slow and not requires_model"

step "Models and data"
read -r STUDENT TEACHER < <(python - <<'PY'
from huggingface_hub import snapshot_download
pins = {"Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
        "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"}
print(*(snapshot_download(r, revision=s) for r, s in pins.items()))
PY
)
DATA=$OUT/train.jsonl
python tests/gpu/checks.py make-data "$DATA"
CT=configs/chat_template_qwen3_generation.jinja
python -m scripts.causal_lm.verify_dataloader --tokenizer "$STUDENT" --data "$DATA" --chat_template "$CT"

TR="torchrun --standalone --nproc_per_node 1"
COMMON="--data_path $DATA --chat_template_path $CT --per_device_train_batch_size 1 --gbs 2 --logging_steps 1 --seed 1234"
D="--module scripts.causal_lm.distill_fsdp2 --student_path $STUDENT --teacher_path $TEACHER $COMMON"
L="--module scripts.causal_lm.lora_fsdp2 --model_path $STUDENT $COMMON"

step "Global batch size that does not divide is rejected before loading"
if $TR $L --output_dir "$OUT/gbs-bad" --max_steps 1 --per_device_train_batch_size 2 --gbs 3; then echo "expected failure"; exit 1; fi

for kind in distill lora; do
  [ "$kind" = distill ] && A=$D || A=$L
  step "$kind: train 2 steps, saving each step"
  $TR $A --output_dir "$OUT/$kind" --max_steps 2 --save_steps 1
  test -d "$OUT/$kind/dcp/step_1"
  step "$kind: resume from step 1 and finish"
  $TR $A --output_dir "$OUT/$kind-resumed" --max_steps 2 --dcp_resume_path "$OUT/$kind/dcp/step_1"
  step "$kind: resume at step 1 with no updates matches a fresh 1-step run"
  $TR $A --output_dir "$OUT/$kind-rt" --max_steps 1 --dcp_resume_path "$OUT/$kind/dcp/step_1"
  $TR $A --output_dir "$OUT/$kind-fresh1" --max_steps 1
  python tests/gpu/checks.py same "$OUT/$kind-rt" "$OUT/$kind-fresh1"
done

step "Exports reload"
python tests/gpu/checks.py reload-full "$OUT/distill-resumed" "$STUDENT"
python tests/gpu/checks.py reload-lora "$OUT/lora-resumed" "$STUDENT"

echo; echo "smoke: ok"
