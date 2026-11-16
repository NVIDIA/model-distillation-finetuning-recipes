#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a decision-model checkpoint with one native inference worker per GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.decision_model.checkpoint_manifest import file_sha256, weight_manifest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for number, text in enumerate(handle, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON ({error})") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: row must be a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def validate_ids(rows: list[dict[str, Any]], name: str) -> None:
    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(f"{name} IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} IDs must be unique")


def parse_gpus(value: str) -> list[str]:
    gpus = [part.strip() for part in value.split(",")]
    if not gpus or any(not part.isdecimal() for part in gpus):
        raise ValueError("--gpus must be comma-separated non-negative GPU indices")
    gpus = [str(int(part)) for part in gpus]
    if len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must name each GPU only once")
    return gpus


def shard_rows(rows: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    if workers < 1:
        raise ValueError("at least one worker is required")
    return [rows[index::workers] for index in range(workers)]


def merge_predictions(
    source_rows: list[dict[str, Any]],
    prediction_paths: list[Path],
    *,
    allow_partial: bool = False,
) -> list[dict[str, Any]]:
    validate_ids(source_rows, "input")
    by_id: dict[str, dict[str, Any]] = {}
    for path in prediction_paths:
        predictions = read_jsonl(path)
        validate_ids(predictions, str(path))
        for prediction in predictions:
            row_id = prediction["id"]
            if row_id in by_id:
                raise ValueError(f"duplicate prediction ID {row_id!r}")
            by_id[row_id] = prediction
    source_ids = {row["id"] for row in source_rows}
    extra = set(by_id) - source_ids
    if extra:
        raise ValueError(f"{len(extra)} predictions do not correspond to input rows")
    missing = source_ids - set(by_id)
    if missing and not allow_partial:
        raise ValueError(f"missing predictions for {len(missing)} input rows")
    return [by_id[row["id"]] for row in source_rows if row["id"] in by_id]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-module", type=Path)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--gpus", default="0", help="comma-separated physical GPU indices")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    try:
        args.gpus = parse_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))
    return args


def run_worker(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from scripts.decision_model.runtime import (
        DecisionRows,
        Scorer,
        batched_option_logits,
        enable_bidirectional,
        resolve_prompt_module,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for decision-model inference")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    pretrained_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.model_revision:
        pretrained_kwargs["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, **pretrained_kwargs)
    prompt_module = resolve_prompt_module(args.checkpoint, args.prompt_module)
    prepare = DecisionRows(tokenizer, prompt_module, args.max_tokens)
    model = AutoModel.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        **pretrained_kwargs,
    ).to(device)
    enable_bidirectional(model)
    scorer = Scorer(model).eval()
    predictions = []
    failures = []
    rows = read_jsonl(args.input)
    validate_ids(rows, "input")
    with torch.inference_mode():
        for row in rows:
            try:
                example = prepare(row)
                if example is None:
                    raise ValueError(f"prompt exceeds --max-tokens {args.max_tokens}")
                logits = batched_option_logits(
                    [example], scorer, mask_id=int(model.mask_token_id), device=device
                )[0]
                probabilities = logits.softmax(-1).cpu().tolist()
                labels = example["labels"]
                if len(probabilities) != len(labels) or not all(
                    math.isfinite(value) for value in probabilities
                ):
                    raise ValueError("model produced invalid option probabilities")
                predictions.append(
                    {
                        "id": row["id"],
                        "probs": dict(zip(labels, probabilities, strict=True)),
                        "pred": labels[max(range(len(labels)), key=probabilities.__getitem__)],
                    }
                )
            except Exception as error:
                if not args.allow_partial:
                    raise RuntimeError(
                        f"inference failed for row {row['id']!r}: {error}"
                    ) from error
                failures.append({"id": row["id"], "error": str(error)})
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_jsonl(args.output_dir / "failures.jsonl", failures)
    summary = {
        "input_rows": len(rows),
        "predicted_rows": len(predictions),
        "failures": len(failures),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.decision_model.runtime import resolve_prompt_module

    checkpoint = args.checkpoint.resolve()
    input_path = args.input.resolve()
    for required in (checkpoint / "config.json", input_path):
        if not required.is_file():
            raise ValueError(f"required path does not exist: {required}")
    prompt_module = resolve_prompt_module(checkpoint, args.prompt_module)
    input_sha256 = file_sha256(input_path)
    prompt_module_sha256 = file_sha256(prompt_module)
    rows = read_jsonl(input_path)
    validate_ids(rows, "input")
    if len(rows) < len(args.gpus):
        raise ValueError(f"{len(rows)} rows cannot be split across {len(args.gpus)} GPUs")
    manifest = weight_manifest(checkpoint)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shards_dir = output / "shards"
    logs_dir = output / "logs"
    shards_dir.mkdir()
    logs_dir.mkdir()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    processes = []
    log_handles = []
    prediction_paths = []
    summary_paths = []
    shards = shard_rows(rows, len(args.gpus))
    try:
        for worker, (gpu, shard) in enumerate(zip(args.gpus, shards, strict=True)):
            shard_path = shards_dir / f"input-{worker:02d}.jsonl"
            worker_dir = shards_dir / f"worker-{worker:02d}"
            worker_dir.mkdir()
            write_jsonl(shard_path, shard)
            command = [
                sys.executable,
                "-m",
                "scripts.decision_model.evaluate",
                "--worker",
                "--checkpoint",
                str(checkpoint),
                "--input",
                str(shard_path),
                "--output-dir",
                str(worker_dir),
                "--prompt-module",
                str(prompt_module),
                "--max-tokens",
                str(args.max_tokens),
                "--attn-implementation",
                args.attn_implementation,
            ]
            if args.model_revision:
                command.extend(["--model-revision", args.model_revision])
            if args.allow_partial:
                command.append("--allow-partial")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.setdefault("TOKENIZERS_PARALLELISM", "false")
            stdout = (logs_dir / f"worker-{worker:02d}.stdout.log").open("wb")
            log_handles.append(stdout)
            stderr = (logs_dir / f"worker-{worker:02d}.stderr.log").open("wb")
            log_handles.append(stderr)
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=stdout,
                stderr=stderr,
            )
            processes.append((worker, gpu, process))
            prediction_paths.append(worker_dir / "predictions.jsonl")
            summary_paths.append(worker_dir / "summary.json")
        pending = processes.copy()
        while pending:
            running = []
            for worker, gpu, process in pending:
                code = process.poll()
                if code is None:
                    running.append((worker, gpu, process))
                elif code:
                    raise RuntimeError(
                        f"worker {worker}/GPU {gpu} failed with exit code {code}; inspect {logs_dir}"
                    )
            pending = running
            if pending:
                time.sleep(0.1)
    finally:
        for _, _, process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for handle in log_handles:
            handle.close()
    predictions = merge_predictions(rows, prediction_paths, allow_partial=args.allow_partial)
    summaries = [json.loads(path.read_text()) for path in summary_paths]
    failures = sum(int(summary["failures"]) for summary in summaries)
    if failures and not args.allow_partial:
        raise ValueError(f"evaluation reported {failures} failed rows")
    if not predictions:
        raise ValueError("evaluation produced no predictions")
    for name, path, digest in (
        ("input", input_path, input_sha256),
        ("prompt module", prompt_module, prompt_module_sha256),
    ):
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"{name} changed during evaluation: {path}")
    predictions_path = output / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    elapsed = time.monotonic() - started
    metadata = {
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "input": str(input_path),
        "input_sha256": input_sha256,
        "input_rows": len(rows),
        "predicted_rows": len(predictions),
        "evaluator_failures": failures,
        "allow_partial": args.allow_partial,
        "throughput_rows_per_second": len(predictions) / elapsed,
        "checkpoint": str(checkpoint),
        "model_revision": args.model_revision or None,
        "model_sha256": manifest["sha256"],
        "model_bytes": manifest["bytes"],
        "weight_files": manifest["files"],
        "index_sha256": manifest.get("index_sha256"),
        "prompt_module": str(prompt_module),
        "prompt_module_sha256": prompt_module_sha256,
        "max_tokens": args.max_tokens,
        "attention_implementation": args.attn_implementation,
        "precision": "bfloat16",
        "gpus": args.gpus,
        "shard_rows": [len(shard) for shard in shards],
        "predictions": str(predictions_path),
        "predictions_sha256": file_sha256(predictions_path),
        "peak_allocated_gib": max(summary["peak_allocated_gib"] for summary in summaries),
        "shard_summaries": summaries,
        "evaluator_sha256": file_sha256(Path(__file__).resolve()),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker:
        run_worker(args)
    else:
        print(json.dumps(run_evaluation(args), indent=2))


if __name__ == "__main__":
    main()
