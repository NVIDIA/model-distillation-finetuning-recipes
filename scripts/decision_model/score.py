#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score decision-model probabilities against a labeled reference JSONL file."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.decision_model.checkpoint_manifest import file_sha256

RUN_METADATA_FIELDS = (
    "started_at",
    "completed_at",
    "elapsed_seconds",
    "input",
    "input_sha256",
    "input_rows",
    "predicted_rows",
    "evaluator_failures",
    "throughput_rows_per_second",
    "checkpoint",
    "model_sha256",
    "model_revision",
    "prompt_module",
    "prompt_module_sha256",
    "attention_implementation",
    "precision",
    "max_tokens",
    "peak_allocated_gib",
    "evaluator_sha256",
    "gpus",
    "predictions",
    "predictions_sha256",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, text in enumerate(handle, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({error})") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    return rows


def index_unique(rows: list[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"{name} row has an invalid id: {row_id!r}")
        if row_id in result:
            raise ValueError(f"{name} has duplicate id {row_id!r}")
        result[row_id] = row
    return result


def normalized_probabilities(prediction: dict[str, Any], labels: list[str]) -> dict[str, float]:
    raw = prediction.get("probs")
    if not isinstance(raw, dict):
        raise ValueError(f"prediction {prediction.get('id')!r} has no probability mapping")
    if set(raw) != set(labels):
        raise ValueError(
            f"prediction {prediction.get('id')!r} probability labels do not match the taxonomy"
        )
    probabilities = {}
    for label in labels:
        try:
            value = float(raw[label])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"prediction {prediction.get('id')!r} has non-numeric probability for {label!r}"
            ) from error
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"prediction {prediction.get('id')!r} has invalid probability for {label!r}"
            )
        probabilities[label] = value
    scale = max(probabilities.values())
    if scale <= 0:
        raise ValueError(f"prediction {prediction.get('id')!r} has zero probability mass")
    scaled = {label: value / scale for label, value in probabilities.items()}
    total = math.fsum(scaled.values())
    return {label: value / total for label, value in scaled.items()}


def target_of(reference: dict[str, Any]) -> str:
    section = reference.get("reference")
    if not isinstance(section, dict) or not isinstance(section.get("target"), str):
        raise ValueError(f"reference {reference.get('id')!r} lacks reference.target")
    return section["target"]


def score_target(
    references: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    label_set = set(labels)
    records = []
    missing_ids = []
    for reference in references:
        row_id = reference["id"]
        target = target_of(reference)
        if target not in label_set:
            raise ValueError(f"reference {row_id!r} has unknown target {target!r}")
        prediction = predictions_by_id.get(row_id)
        if prediction is None:
            missing_ids.append(row_id)
            continue
        probabilities = normalized_probabilities(prediction, labels)
        if "pred" in prediction:
            predicted = prediction["pred"]
            if (
                not isinstance(predicted, str)
                or predicted not in label_set
                or probabilities[predicted] != max(probabilities.values())
            ):
                raise ValueError(
                    f"prediction {row_id!r} pred must be a maximum-probability taxonomy label"
                )
        else:
            predicted = max(labels, key=probabilities.__getitem__)
        confidence = probabilities[predicted]
        records.append(
            {
                "id": row_id,
                "target": target,
                "predicted": predicted,
                "correct": predicted == target,
                "confidence": confidence,
                "p_target": probabilities[target],
                "brier": sum(
                    (probabilities[label] - float(label == target)) ** 2 for label in labels
                ),
            }
        )
    if not records:
        raise ValueError("no predictions matched the references")

    support = Counter(record["target"] for record in records)
    predicted_count = Counter(record["predicted"] for record in records)
    true_positive = Counter(record["target"] for record in records if record["correct"])
    per_class = {}
    for label in labels:
        tp = true_positive[label]
        class_support = support[label]
        predicted_support = predicted_count[label]
        precision = tp / predicted_support if predicted_support else 0.0
        recall = tp / class_support if class_support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": class_support,
            "predicted": predicted_support,
            "correct": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    supported = [label for label in labels if support[label]]
    total = len(records)
    accuracy = sum(record["correct"] for record in records) / total
    # Use every declared class, including classes predicted spuriously when
    # they have no reference support. This matches an explicit taxonomy-wide F1.
    macro_f1 = sum(per_class[label]["f1"] for label in labels) / len(labels)
    balanced_accuracy = sum(per_class[label]["recall"] for label in supported) / len(supported)
    weighted_f1 = sum(per_class[label]["f1"] * support[label] for label in supported) / total
    log_loss = sum(-math.log(max(1e-12, record["p_target"])) for record in records) / total
    brier = sum(record["brier"] for record in records) / total
    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        bins[min(9, int(record["confidence"] * 10))].append(record)
    ece = sum(
        len(bin_records)
        / total
        * abs(
            sum(record["correct"] for record in bin_records) / len(bin_records)
            - sum(record["confidence"] for record in bin_records) / len(bin_records)
        )
        for bin_records in bins.values()
    )
    confusion = Counter(
        (record["target"], record["predicted"]) for record in records if not record["correct"]
    )
    top_confusions = [
        {"target": target, "predicted": predicted, "count": count}
        for (target, predicted), count in confusion.most_common(15)
    ]
    return {
        "expected_rows": len(references),
        "scored_rows": total,
        "missing_predictions": len(missing_ids),
        "missing_prediction_ids": missing_ids[:100],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_f1_labels": "all taxonomy labels",
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "log_loss": log_loss,
        "brier": brier,
        "ece_10_bin": ece,
        "supported_classes": len(supported),
        "per_class": per_class,
        "top_confusions": top_confusions,
    }


def score_all(
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    labels: list[str],
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    if (
        not isinstance(labels, list)
        or len(labels) < 2
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len(labels) != len(set(labels))
    ):
        raise ValueError("taxonomy must contain at least two unique non-empty string labels")
    references_by_id = index_unique(references, name="references")
    predictions_by_id = index_unique(predictions, name="predictions")
    extra_predictions = sorted(set(predictions_by_id) - set(references_by_id))
    if extra_predictions:
        raise ValueError(f"{len(extra_predictions)} predictions do not correspond to references")
    missing = set(references_by_id) - set(predictions_by_id)
    if missing and not allow_partial:
        raise ValueError(f"missing predictions for {len(missing)} reference rows")
    metrics = score_target(references, predictions_by_id, labels)
    return {
        "reference_rows": len(references),
        "prediction_rows": len(predictions),
        "extra_predictions": len(extra_predictions),
        "extra_prediction_ids": extra_predictions[:100],
        "allow_partial": allow_partial,
        "metrics": metrics,
    }


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def markdown_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    run = result.get("run", {})
    weakest = sorted(
        (
            (label, class_metrics)
            for label, class_metrics in metrics["per_class"].items()
            if class_metrics["support"]
        ),
        key=lambda item: (item[1]["recall"], -item[1]["support"], item[0]),
    )[:8]
    lines = [
        "# Decision model evaluation",
        "",
        f"Evaluation target: {result.get('target', 'reference labels')}.",
        "",
        "## Results",
        "",
        f"- Accuracy: **{percent(metrics['accuracy'])}**",
        f"- Macro F1 (all taxonomy labels): **{percent(metrics['macro_f1'])}**",
        f"- Weighted F1: {percent(metrics['weighted_f1'])}",
        f"- Balanced accuracy: {percent(metrics['balanced_accuracy'])}",
        f"- Log loss: {metrics['log_loss']:.4f}",
        f"- Multiclass Brier score: {metrics['brier']:.4f}",
        f"- 10-bin ECE: {metrics['ece_10_bin']:.4f}",
        (
            f"- Coverage: {metrics['scored_rows']}/{metrics['expected_rows']} "
            f"({metrics['missing_predictions']} missing)"
        ),
        "",
        "## Weakest reference classes by recall",
        "",
    ]
    for label, metrics in weakest:
        lines.append(
            f"- {label}: {percent(metrics['recall'])} recall "
            f"({metrics['correct']}/{metrics['support']})"
        )
    if run:
        lines.extend(
            [
                "",
                "## Run provenance",
                "",
                f"- Model checkpoint: `{run.get('checkpoint', 'unknown')}`",
                f"- Model SHA-256: `{run.get('model_sha256', 'unknown')}`",
                f"- Prompt module SHA-256: `{run.get('prompt_module_sha256', 'unknown')}`",
                f"- Input SHA-256: `{run.get('input_sha256', 'unknown')}`",
                f"- GPUs: {', '.join(run.get('gpus', []))}",
                f"- Elapsed: {run.get('elapsed_seconds', 0):.2f} seconds",
                f"- Throughput: {run.get('throughput_rows_per_second', 0):.2f} rows/second",
                f"- Evaluator failures: {run.get('evaluator_failures', 0)}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def validate_run_metadata(
    metadata: dict[str, Any],
    references: Path,
    predictions: Path,
    reference_rows: int,
    prediction_rows: int,
    *,
    allow_partial: bool,
) -> None:
    expected = {
        "input_sha256": file_sha256(references),
        "predictions_sha256": file_sha256(predictions),
        "input_rows": reference_rows,
        "predicted_rows": prediction_rows,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"run metadata {field} does not match the scored inputs")
    failures = metadata.get("evaluator_failures")
    if not isinstance(failures, int) or isinstance(failures, bool) or failures < 0:
        raise ValueError("run metadata must record a non-negative evaluator_failures count")
    if failures and not allow_partial:
        raise ValueError(f"run metadata reports {failures} failed evaluation rows")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        required=True,
    )
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--target-name", default="reference labels")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = [args.output_json.resolve(), args.output_markdown.resolve()]
    inputs = {args.references.resolve(), args.predictions.resolve(), args.taxonomy.resolve()}
    if args.run_metadata:
        inputs.add(args.run_metadata.resolve())
    if outputs[0] == outputs[1] or inputs.intersection(outputs):
        parser.error("output files must be distinct from each other and from input files")

    references = read_jsonl(args.references)
    predictions = read_jsonl(args.predictions)
    taxonomy = json.loads(args.taxonomy.read_text())
    if not isinstance(taxonomy, dict) or not isinstance(taxonomy.get("labels"), dict):
        raise ValueError("taxonomy must be an object containing a labels mapping")
    labels = list(taxonomy["labels"])
    result = score_all(references, predictions, labels, allow_partial=args.allow_partial)
    result["target"] = args.target_name
    result["references"] = str(args.references.resolve())
    result["predictions"] = str(args.predictions.resolve())
    result["taxonomy"] = str(args.taxonomy.resolve())
    if args.run_metadata:
        full_run = json.loads(args.run_metadata.read_text())
        if not isinstance(full_run, dict):
            raise ValueError("run metadata must be a JSON object")
        validate_run_metadata(
            full_run,
            args.references,
            args.predictions,
            len(references),
            len(predictions),
            allow_partial=args.allow_partial,
        )
        result["run"] = {
            field: full_run[field] for field in RUN_METADATA_FIELDS if field in full_run
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    report = markdown_report(result)
    args.output_markdown.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
