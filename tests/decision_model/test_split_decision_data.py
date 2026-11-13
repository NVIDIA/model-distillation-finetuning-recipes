# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.decision_model import split_data as split_decision_data

REPO = Path(__file__).resolve().parents[2]


class SplitDecisionDataTest(unittest.TestCase):
    def write_rows(self, path: Path) -> None:
        rows = []
        for label, count in (("common", 40), ("rare", 3)):
            for index in range(count):
                rows.append(
                    {
                        "id": f"{label}-{index}",
                        "input": {"state": {"headline": f"{label} {index}"}},
                        "reference": {"target": label},
                    }
                )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_split_is_disjoint_stratified_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            self.write_rows(source)

            first_train = root / "first-train.jsonl"
            first_validation = root / "first-validation.jsonl"
            first_test = root / "first-test.jsonl"
            summary = split_decision_data.split_file(
                source,
                first_train,
                first_validation,
                first_test,
                validation_fraction=0.2,
                test_fraction=0.1,
                seed=101,
            )
            second_train = root / "second-train.jsonl"
            second_validation = root / "second-validation.jsonl"
            second_test = root / "second-test.jsonl"
            split_decision_data.split_file(
                source,
                second_train,
                second_validation,
                second_test,
                validation_fraction=0.2,
                test_fraction=0.1,
                seed=101,
            )

            train_rows = [json.loads(line) for line in first_train.read_text().splitlines()]
            validation_rows = [
                json.loads(line) for line in first_validation.read_text().splitlines()
            ]
            test_rows = [json.loads(line) for line in first_test.read_text().splitlines()]
            train_ids = {row["id"] for row in train_rows}
            validation_ids = {row["id"] for row in validation_rows}
            test_ids = {row["id"] for row in test_rows}

            self.assertFalse(train_ids & validation_ids)
            self.assertFalse(train_ids & test_ids)
            self.assertFalse(validation_ids & test_ids)
            self.assertEqual(len(train_ids | validation_ids | test_ids), 43)
            expected_counts = {
                "train": {"common": 28, "rare": 1},
                "validation": {"common": 8, "rare": 1},
                "test": {"common": 4, "rare": 1},
            }
            for split, rows in (
                ("train", train_rows),
                ("validation", validation_rows),
                ("test", test_rows),
            ):
                self.assertEqual({row["reference"]["target"] for row in rows}, {"common", "rare"})
                self.assertEqual(summary[f"{split}_label_counts"], expected_counts[split])
                self.assertEqual(summary[f"{split}_rows"], len(rows))
            self.assertTrue(summary["stratified"])
            self.assertEqual(first_train.read_bytes(), second_train.read_bytes())
            self.assertEqual(first_validation.read_bytes(), second_validation.read_bytes())
            self.assertEqual(first_test.read_bytes(), second_test.read_bytes())

    def test_cli_writes_three_splits_and_verified_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "decisions.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.decision_model.convert_chat",
                    "--input",
                    str(REPO / "examples/decision_model/chat.jsonl"),
                    "--taxonomy",
                    str(REPO / "examples/decision_model/taxonomy.json"),
                    "--output",
                    str(source),
                ],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            )
            paths = {split: root / f"{split}.jsonl" for split in ("train", "validation", "test")}
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.decision_model.split_data",
                    "--input",
                    str(source),
                    "--train-output",
                    str(paths["train"]),
                    "--validation-output",
                    str(paths["validation"]),
                    "--test-output",
                    str(paths["test"]),
                    "--validation-fraction",
                    "0.05",
                    "--test-fraction",
                    "0.05",
                    "--seed",
                    "101",
                ],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertTrue(summary["stratified"])
            for split, expected_rows in (("train", 12), ("validation", 4), ("test", 4)):
                self.assertEqual(summary[f"{split}_rows"], expected_rows)
                self.assertEqual(
                    set(summary[f"{split}_label_counts"]), set(summary["label_counts"])
                )
                self.assertEqual(
                    summary[f"{split}_sha256"],
                    hashlib.sha256(paths[split].read_bytes()).hexdigest(),
                )
            metadata = paths["validation"].with_suffix(".jsonl.metadata.json")
            self.assertEqual(json.loads(metadata.read_text()), summary)

    def test_rejects_labels_with_too_few_examples_for_three_splits(self):
        for count in (1, 2):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.jsonl"
                self.write_rows(source)
                with source.open("a") as handle:
                    for index in range(count):
                        handle.write(
                            json.dumps({"id": f"scarce-{index}", "reference": {"target": "scarce"}})
                            + "\n"
                        )
                with self.assertRaisesRegex(ValueError, "'scarce'.*at least three"):
                    split_decision_data.split_file(
                        source, root / "train.jsonl", root / "validation.jsonl", root / "test.jsonl"
                    )
                self.assertEqual(list(root.iterdir()), [source])

    def test_rejects_invalid_holdout_fractions(self):
        for validation, test in ((0, 0.05), (0.05, 0), (0.6, 0.4), (0.9, 0.2)):
            with (
                self.subTest(validation=validation, test=test),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = root / "source.jsonl"
                self.write_rows(source)
                with self.assertRaisesRegex(ValueError, "fraction"):
                    split_decision_data.split_file(
                        source,
                        root / "train.jsonl",
                        root / "validation.jsonl",
                        root / "test.jsonl",
                        validation_fraction=validation,
                        test_fraction=test,
                    )
                self.assertEqual(list(root.iterdir()), [source])

    def test_stratification_failure_rejects_outputs_even_when_all_labels_are_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            self.write_rows(source)
            with (
                patch.object(
                    split_decision_data,
                    "_select_holdout_ids",
                    return_value=({"common-0", "rare-0"}, {"common-1", "rare-1"}),
                ),
                self.assertRaisesRegex(ValueError, "failed stratification"),
            ):
                split_decision_data.split_file(
                    source, root / "train.jsonl", root / "validation.jsonl", root / "test.jsonl"
                )
            self.assertEqual(list(root.iterdir()), [source])

    def test_overlapping_holdouts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            self.write_rows(source)
            with (
                patch.object(
                    split_decision_data,
                    "_select_holdout_ids",
                    return_value=({"common-0"}, {"common-0"}),
                ),
                self.assertRaisesRegex(ValueError, "must not overlap"),
            ):
                split_decision_data.split_file(
                    source, root / "train.jsonl", root / "validation.jsonl", root / "test.jsonl"
                )
            self.assertEqual(list(root.iterdir()), [source])

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "same", "reference": {"target": "first"}}),
                        json.dumps({"id": "same", "reference": {"target": "second"}}),
                    ]
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                split_decision_data.split_file(
                    source,
                    root / "train.jsonl",
                    root / "validation.jsonl",
                    root / "test.jsonl",
                )

    def test_metadata_output_cannot_overwrite_input_or_split_outputs(self):
        for protected in ("input", "train", "test"):
            with self.subTest(protected=protected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                validation = root / "validation.jsonl"
                metadata = validation.with_suffix(".jsonl.metadata.json")
                source = metadata if protected == "input" else root / "input.jsonl"
                train = metadata if protected == "train" else root / "train.jsonl"
                test = metadata if protected == "test" else root / "test.jsonl"
                source.write_text("protected source")
                with self.assertRaisesRegex(ValueError, "must be distinct"):
                    split_decision_data.split_file(source, train, validation, test)
                self.assertEqual(source.read_text(), "protected source")
                self.assertFalse(validation.exists())

    def test_source_change_between_passes_rejects_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            self.write_rows(source)
            select = split_decision_data._select_holdout_ids

            def change_after_selection(*args):
                selected = select(*args)
                source.write_text(source.read_text() + "\n")
                return selected

            with (
                patch.object(split_decision_data, "_select_holdout_ids", change_after_selection),
                self.assertRaisesRegex(ValueError, "input changed during splitting"),
            ):
                split_decision_data.split_file(
                    source,
                    root / "train.jsonl",
                    root / "validation.jsonl",
                    root / "test.jsonl",
                )
            self.assertFalse((root / "train.jsonl").exists())
            self.assertFalse((root / "validation.jsonl").exists())
            self.assertFalse((root / "test.jsonl").exists())
            self.assertFalse(list(root.glob(".*.tmp-*")))

    def test_output_creation_failure_cleans_existing_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            self.write_rows(source)
            invalid_parent = root / "not-a-directory"
            invalid_parent.write_text("keep")
            with self.assertRaises(FileExistsError):
                split_decision_data.split_file(
                    source,
                    root / "train.jsonl",
                    root / "validation.jsonl",
                    invalid_parent / "test.jsonl",
                )
            self.assertFalse((root / "train.jsonl").exists())
            self.assertFalse(list(root.glob(".*.tmp-*")))


if __name__ == "__main__":
    unittest.main()
