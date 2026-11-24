# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks used by tests/gpu/smoke.sh. Each command exits non-zero on failure."""

import glob
import json
import os
import sys

import torch
from safetensors.torch import load_file


def load_dir(path):
    state = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        state.update(load_file(f))
    if not state:
        sys.exit(f"no safetensors in {path}")
    return state


def make_data(path, rows=32):
    topics = ["FSDP", "LoRA", "distillation", "gradient accumulation", "checkpointing"]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(rows):
            t = topics[i % len(topics)]
            msgs = [{"role": "user", "content": f"Explain {t} briefly (example {i})."},
                    {"role": "assistant", "content": f"{t}: point {i % 7} relates to {topics[(i + 1) % 5]}."}]
            f.write(json.dumps({"messages": msgs}) + "\n")


def same(a, b):
    sa, sb = load_dir(a), load_dir(b)
    if set(sa) != set(sb):
        sys.exit(f"different tensor names: {sorted(set(sa) ^ set(sb))[:5]}")
    diff = [k for k in sa if not torch.equal(sa[k], sb[k])]
    if diff:
        sys.exit(f"{len(diff)}/{len(sa)} tensors differ, e.g. {diff[:3]}")
    print(f"identical: {len(sa)} tensors")


def reload_full(out, base):
    from transformers import AutoModelForCausalLM
    model, info = AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.bfloat16, output_loading_info=True)
    bad = {k: v for k, v in info.items() if k in ("missing_keys", "unexpected_keys", "mismatched_keys") and v}
    if bad:
        sys.exit(f"export does not load cleanly: {bad}")
    ref = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).state_dict()
    sd = model.state_dict()
    if not all(torch.isfinite(v.float()).all() for v in sd.values()):
        sys.exit("export has non-finite weights")
    if all(torch.equal(sd[k], ref[k]) for k in sd):
        sys.exit("export equals the base model: training changed nothing")
    print(f"reloaded: {len(sd)} tensors, changed from base")


def reload_lora(out, base):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    state = load_file(os.path.join(out, "adapter_model.safetensors"))
    b = [k for k in state if "lora_B" in k]
    if not b or not all(state[k].abs().max() > 0 for k in b):
        sys.exit("adapter has untrained (all-zero) lora_B tensors")
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32), out)
    loaded = {k.replace(".default", ""): v for k, v in model.state_dict().items()}
    missing = [k for k in state if k not in loaded or not torch.equal(loaded[k].to(state[k].dtype), state[k])]
    if missing:
        sys.exit(f"{len(missing)}/{len(state)} adapter tensors did not load, e.g. {missing[:3]}")
    print(f"adapter reloaded: {len(state)} tensors")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"make-data": make_data, "same": same, "reload-full": reload_full, "reload-lora": reload_lora}[cmd](*args)
