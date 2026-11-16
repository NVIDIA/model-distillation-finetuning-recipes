# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.decision_model import convert_chat as converter

REPO = Path(__file__).resolve().parents[2]


class ConvertChatToDecisionTest(unittest.TestCase):
    def test_public_cli_converts_chat_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "chat.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "messages": [
                            {"role": "user", "content": "Service unavailable."},
                            {"role": "assistant", "content": "[[outage]]"},
                        ],
                    }
                )
                + "\n"
            )
            taxonomy = root / "taxonomy.json"
            taxonomy.write_text(
                json.dumps(
                    {
                        "question": "Which route?",
                        "labels": {
                            "outage": "Service outage",
                            "other": "None of these",
                        },
                    }
                )
            )
            output = root / "decisions.jsonl"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.decision_model.convert_chat",
                    "--input",
                    str(source),
                    "--taxonomy",
                    str(taxonomy),
                    "--output",
                    str(output),
                ],
                cwd=REPO,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            row = json.loads(output.read_text())
            self.assertEqual(row["reference"]["target"], "outage")
            self.assertEqual(row["input"]["state"], "Service unavailable.")
            self.assertTrue(output.with_suffix(".jsonl.metadata.json").is_file())

    def test_prepare_file_converts_and_verifies_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = {"A": "class A", "B": "class B"}
            taxonomy = root / "taxonomy.json"
            taxonomy.write_text(json.dumps({"question": "Choose.", "labels": labels}))
            source_rows = [
                {
                    "id": "one",
                    "headline": "First headline",
                    "source_index": 4,
                    "teacher_raw_output": "[[A]]",
                    "messages": [{"role": "assistant", "content": "[[A]]"}],
                },
                {
                    "id": "two",
                    "headline": "Second headline",
                    "source_index": 8,
                    "teacher_raw_output": "[[B]]",
                    "messages": [{"role": "assistant", "content": "[[B]]"}],
                },
            ]
            source = root / "validation.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "valid_labels": list(labels),
                        "splits": {
                            "validation": {
                                "completed_rows": 2,
                                "sha256": source_sha,
                            }
                        },
                    }
                )
            )
            output = root / "converted.jsonl"

            summary = converter.prepare_file(
                source,
                manifest,
                taxonomy,
                output,
                state_field="headline",
                raw_label_field="teacher_raw_output",
            )

            converted = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(converted[0]["reference"]["target"], "A")
            self.assertNotIn("original_reference", converted[0])
            question = converted[0]["input"]["questions"]["event_type"]
            self.assertEqual(question["criteria"], labels)
            self.assertEqual(converted[0]["input"]["state"], {"headline": "First headline"})

    def test_reference_label_rejects_disagreement(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            converter.reference_label(
                {
                    "teacher_raw_output": "[[A]]",
                    "messages": [{"role": "assistant", "content": "[[B]]"}],
                },
                raw_label_field="teacher_raw_output",
            )
        with self.assertRaisesRegex(ValueError, "missing 'teacher_raw_output'"):
            converter.reference_label(
                {"messages": [{"role": "assistant", "content": "[[A]]"}]},
                raw_label_field="teacher_raw_output",
            )

    def test_generic_chat_uses_last_user_message(self):
        converted = converter.convert_row(
            {
                "messages": [
                    {"role": "system", "content": "Classify the request."},
                    {"role": "user", "content": "The service is unavailable."},
                    {"role": "assistant", "content": "[[outage]]"},
                ]
            },
            question="Which route?",
            labels={"outage": "Service is unavailable", "other": "None of these"},
            fallback_id="row-1",
        )

        self.assertEqual(converted["id"], "row-1")
        self.assertEqual(
            converted["input"]["state"],
            "The service is unavailable.",
        )
        self.assertEqual(converted["reference"]["target"], "outage")

    def test_soft_labels_are_validated_normalized_and_ordered(self):
        converted = converter.convert_row(
            {
                "messages": [
                    {"role": "user", "content": "The service is unavailable."},
                    {"role": "assistant", "content": "[[outage]]"},
                ],
                "soft_labels": {"other": 1, "outage": 3},
            },
            question="Which route?",
            labels={"outage": "Service is unavailable", "other": "None of these"},
            state_field=None,
            raw_label_field=None,
            fallback_id="row-1",
        )

        self.assertEqual(
            converted["soft_labels"],
            {"outage": 0.75, "other": 0.25},
        )

    def test_malformed_messages_and_empty_labels_have_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "final message must be an object"):
            converter.reference_label({"messages": ["[[outage]]"]})
        with self.assertRaisesRegex(ValueError, "empty label"):
            converter.parse_wrapped_label("[[   ]]", field="assistant content")
        with self.assertRaisesRegex(ValueError, "not a double-bracketed label"):
            converter.parse_wrapped_label("[[first]][[second]]", field="assistant content")

    def test_soft_labels_reject_unknown_or_nonfinite_weights(self):
        base = {
            "messages": [
                {"role": "user", "content": "The service is unavailable."},
                {"role": "assistant", "content": "[[outage]]"},
            ]
        }
        options = {
            "question": "Which route?",
            "labels": {"outage": "Unavailable", "other": "None"},
            "state_field": None,
            "raw_label_field": None,
            "fallback_id": "row-1",
        }
        with self.assertRaisesRegex(ValueError, "unknown label"):
            converter.convert_row(
                {**base, "soft_labels": {"unknown": 1}},
                **options,
            )
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            converter.convert_row(
                {**base, "soft_labels": {"outage": float("inf")}},
                **options,
            )

    def test_input_and_output_must_be_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "must be different"):
                converter.prepare_file(path, None, path, path)

    def test_skip_writes_rejected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            taxonomy = root / "taxonomy.json"
            taxonomy.write_text(
                json.dumps(
                    {
                        "question": "Choose.",
                        "labels": {"A": "class A", "B": "class B"},
                    }
                )
            )
            rows = [
                {
                    "id": "good",
                    "headline": "Good",
                    "teacher_raw_output": "[[A]]",
                    "messages": [{"role": "assistant", "content": "[[A]]"}],
                },
                {
                    "id": "bad",
                    "headline": "Bad",
                    "teacher_raw_output": "[[B]] is not correct",
                    "messages": [{"role": "assistant", "content": "[[B]]"}],
                },
            ]
            source = root / "training.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "valid_labels": ["A", "B"],
                        "splits": {
                            "training": {
                                "completed_rows": 2,
                                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            }
                        },
                    }
                )
            )
            output = root / "training.decision.jsonl"

            summary = converter.prepare_file(
                source,
                manifest,
                taxonomy,
                output,
                state_field="headline",
                raw_label_field="teacher_raw_output",
                on_error="skip",
            )

            self.assertEqual(summary["input_rows"], 2)
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["skipped_rows"], 1)
            rejected = [
                json.loads(line)
                for line in Path(summary["rejected_output"]).read_text().splitlines()
            ]
            self.assertEqual(rejected[0]["id"], "bad")

    def test_large_finite_soft_weights_remain_normalized(self):
        converted = converter.convert_row(
            {
                "id": "one",
                "messages": [
                    {"role": "user", "content": "A request."},
                    {"role": "assistant", "content": "[[first]]"},
                ],
                "soft_labels": {"first": 1e308, "second": 1e308},
            },
            question="Choose.",
            labels={"first": "First", "second": "Second"},
        )
        self.assertEqual(converted["soft_labels"], {"first": 0.5, "second": 0.5})

    def test_skip_rejects_overflowing_soft_weights_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            rows = [
                {
                    "id": identifier,
                    "messages": [
                        {"role": "user", "content": "A request."},
                        {"role": "assistant", "content": "[[first]]"},
                    ],
                    "soft_labels": {"first": weight, "second": 1},
                }
                for identifier, weight in (("overflow", 10**400), ("valid", 1))
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            taxonomy = root / "taxonomy.json"
            taxonomy.write_text(
                json.dumps(
                    {
                        "question": "Choose.",
                        "labels": {"first": "First", "second": "Second"},
                    }
                )
            )
            output = root / "output.jsonl"

            summary = converter.prepare_file(source, None, taxonomy, output, on_error="skip")

            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["skipped_rows"], 1)
            self.assertEqual(json.loads(output.read_text())["id"], "valid")
            rejected = json.loads(Path(summary["rejected_output"]).read_text())
            self.assertEqual(rejected["id"], "overflow")
            self.assertIn("must be numeric", rejected["error"])

    def test_output_sidecars_cannot_overwrite_inputs(self):
        for protected in ("input", "taxonomy", "manifest"):
            with self.subTest(protected=protected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "output.jsonl"
                inputs = {
                    "input": root / "input.jsonl",
                    "taxonomy": root / "taxonomy.json",
                    "manifest": root / "manifest.json",
                }
                inputs[protected] = output.with_suffix(".jsonl.metadata.json")
                inputs[protected].write_text("protected contents")
                with self.assertRaisesRegex(ValueError, "must be distinct"):
                    converter.prepare_file(
                        inputs["input"],
                        inputs["manifest"],
                        inputs["taxonomy"],
                        output,
                    )
                self.assertEqual(inputs[protected].read_text(), "protected contents")
                self.assertFalse(output.exists())

    def test_duplicate_id_is_fatal_after_a_skipped_invalid_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            rows = [
                {
                    "id": "duplicate",
                    "messages": [
                        {"role": "user", "content": "First request."},
                        {"role": "assistant", "content": "[[unknown]]"},
                    ],
                },
                {
                    "id": "duplicate",
                    "messages": [
                        {"role": "user", "content": "Second request."},
                        {"role": "assistant", "content": "[[first]]"},
                    ],
                },
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            taxonomy = root / "taxonomy.json"
            taxonomy.write_text(
                json.dumps(
                    {
                        "question": "Choose.",
                        "labels": {"first": "First", "second": "Second"},
                    }
                )
            )
            output = root / "output.jsonl"
            output.write_text("existing output")
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                converter.prepare_file(source, None, taxonomy, output, on_error="skip")
            self.assertEqual(output.read_text(), "existing output")
            self.assertFalse(list(root.glob(".*.tmp-*")))

    def test_conversion_with_no_valid_rows_keeps_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text("{}\n")
            taxonomy = root / "taxonomy.json"
            taxonomy.write_text(
                json.dumps(
                    {
                        "question": "Choose.",
                        "labels": {"first": "First", "second": "Second"},
                    }
                )
            )
            output = root / "output.jsonl"
            output.write_text("existing output")
            with self.assertRaisesRegex(ValueError, "no valid rows"):
                converter.prepare_file(source, None, taxonomy, output, on_error="skip")
            self.assertEqual(output.read_text(), "existing output")

    def test_empty_user_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "user message must not be empty"):
            converter.convert_row(
                {
                    "id": "one",
                    "messages": [
                        {"role": "user", "content": "   "},
                        {"role": "assistant", "content": "[[first]]"},
                    ],
                },
                question="Choose.",
                labels={"first": "First", "second": "Second"},
            )


if __name__ == "__main__":
    unittest.main()
