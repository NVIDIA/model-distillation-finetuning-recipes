#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify an exported decision-model checkpoint and compare its CUDA readout paths."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from scripts.decision_model.checkpoint_manifest import (
    file_sha256,
    safetensors_inventory,
    weight_manifest,
)


def verify_export_checksums(checkpoint: Path) -> None:
    """Verify exported file hashes when SHA256SUMS is present."""
    path = checkpoint / "SHA256SUMS"
    if not path.is_file():
        return
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        digest, separator, name = line.partition("  ")
        relative = PurePosixPath(name)
        if (
            not separator
            or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
            or not name
            or relative.is_absolute()
            or PureWindowsPath(name).drive
            or "\\" in name
            or ".." in relative.parts
            or relative.as_posix() != name
            or name in seen
            or name == "SHA256SUMS"
        ):
            raise ValueError(f"invalid checksum entry at {path}:{number}")
        seen.add(name)
        target = checkpoint / relative
        if not target.is_file() or file_sha256(target) != digest.lower():
            raise ValueError(f"export checksum mismatch for {name}")
    if not seen:
        raise ValueError(f"empty checksum manifest: {path}")


def verify_receipt(
    checkpoint: Path,
    *,
    method: str | None = None,
    prompt_module: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads((checkpoint / "pure-pytorch-receipt.json").read_text())
    if not isinstance(receipt, dict):
        raise ValueError("checkpoint receipt must be a JSON object")
    if receipt.get("fsdp2") is not True:
        raise ValueError("checkpoint receipt does not identify FSDP2 training")
    if receipt.get("method") not in {"lora", "full"}:
        raise ValueError("checkpoint receipt has an unsupported method")
    if method and receipt["method"] != method:
        raise ValueError(f"checkpoint method {receipt['method']} != expected {method}")
    manifest = weight_manifest(checkpoint)
    for receipt_key, manifest_key in (("model_sha256", "sha256"), ("model_bytes", "bytes")):
        if receipt.get(receipt_key) != manifest[manifest_key]:
            raise ValueError(f"checkpoint {receipt_key} does not match its receipt")
    if "index_sha256" in receipt and receipt["index_sha256"] != manifest.get("index_sha256"):
        raise ValueError("checkpoint index SHA-256 does not match its receipt")
    inventory = safetensors_inventory(checkpoint)
    count = receipt.get("tensor_count")
    if not isinstance(count, int) or isinstance(count, bool) or count != len(inventory):
        raise ValueError("checkpoint tensor count does not match its receipt")
    if (
        prompt_module is not None
        and receipt.get("prompt_module_sha256") is not None
        and file_sha256(prompt_module) != receipt["prompt_module_sha256"]
    ):
        raise ValueError("prompt module SHA-256 does not match the checkpoint receipt")
    verify_export_checksums(checkpoint)
    return receipt, manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--method", choices=("lora", "full"))
    parser.add_argument("--prompt-module", type=Path)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    for name in ("atol", "rtol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name} must be finite and non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from scripts.decision_model.runtime import (
        DecisionRows,
        Scorer,
        batched_option_logits,
        enable_bidirectional,
        resolve_prompt_module,
    )

    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint validation")
    prompt_module = resolve_prompt_module(args.checkpoint, args.prompt_module)
    receipt, _ = verify_receipt(args.checkpoint, method=args.method, prompt_module=prompt_module)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    prepare = DecisionRows(tokenizer, prompt_module, args.max_tokens)
    with args.data.open() as handle:
        first_line = next((line for line in handle if line.strip()), None)
    if first_line is None:
        raise ValueError("validation data contains no rows")
    row = json.loads(first_line)
    example = prepare(row)
    if example is None:
        raise ValueError("first validation prompt exceeds --max-tokens")
    device = torch.device("cuda", 0)
    model = (
        AutoModel.from_pretrained(
            args.checkpoint,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
        )
        .to(device)
        .eval()
    )
    # Compare the standard encoder entrypoint with the batched scorer used by
    # training and evaluation. This one-row check has no padding on either path.
    for layer in model.encoder.layers:
        layer.self_attn.diffusion_lm = True
    inputs = torch.tensor([example["ids"] + [int(model.mask_token_id)]], device=device)
    with torch.inference_mode():
        hidden = model.encoder(
            input_ids=inputs,
            use_cache=False,
            use_causal_mask=False,
            is_training=False,
        ).last_hidden_state
        serving_logits = model.diffusion_head(hidden[:, -1]).float()[0, example["cand"]]
        enable_bidirectional(model)
        shared_logits = batched_option_logits(
            [example], Scorer(model).eval(), mask_id=int(model.mask_token_id), device=device
        )[0]
        torch.testing.assert_close(
            shared_logits,
            serving_logits,
            atol=args.atol,
            rtol=args.rtol,
            msg="standard encoder and shared decision scorer logits differ",
        )
        probabilities = shared_logits.softmax(-1).cpu().tolist()
    if not probabilities or not all(math.isfinite(value) for value in probabilities):
        raise ValueError("checkpoint produced non-finite probabilities")
    labels = example["labels"]
    result = {
        "checkpoint": str(args.checkpoint),
        "method": receipt["method"],
        "tensor_count": receipt["tensor_count"],
        "device": torch.cuda.get_device_name(device),
        "labels": len(labels),
        "gold": labels[example["gold"]],
        "prediction": labels[max(range(len(labels)), key=probabilities.__getitem__)],
        "probability_sum": sum(probabilities),
        "max_logit_difference": float((shared_logits - serving_logits).abs().max()),
        "atol": args.atol,
        "rtol": args.rtol,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
