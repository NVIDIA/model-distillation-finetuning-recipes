# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import unittest

import pytest

from scripts.decision_model import score as score_predictions


class ScorePredictionsTest(unittest.TestCase):
    def test_scores_reference_targets(self):
        labels = ["A", "B", "C"]
        references = [
            {
                "id": "one",
                "reference": {"target": "A"},
            },
            {
                "id": "two",
                "reference": {"target": "B"},
            },
            {
                "id": "three",
                "reference": {"target": "C"},
            },
        ]
        predictions = [
            {"id": "one", "probs": {"A": 0.8, "B": 0.1, "C": 0.1}},
            {"id": "two", "probs": {"A": 0.1, "B": 0.2, "C": 0.7}},
            {"id": "three", "probs": {"A": 0.1, "B": 0.1, "C": 0.8}},
        ]

        result = score_predictions.score_all(references, predictions, labels)

        self.assertAlmostEqual(result["metrics"]["accuracy"], 2 / 3)
        self.assertEqual(result["metrics"]["missing_predictions"], 0)
        self.assertEqual(
            result["metrics"]["per_class"]["C"]["recall"],
            1.0,
        )

    def test_reports_missing_prediction(self):
        labels = ["A", "B"]
        references = [
            {
                "id": "one",
                "reference": {"target": "A"},
            },
            {
                "id": "two",
                "reference": {"target": "B"},
            },
        ]
        predictions = [{"id": "one", "probs": {"A": 0.9, "B": 0.1}}]

        with self.assertRaisesRegex(ValueError, "missing predictions"):
            score_predictions.score_all(references, predictions, labels)
        result = score_predictions.score_all(references, predictions, labels, allow_partial=True)

        self.assertEqual(result["metrics"]["scored_rows"], 1)
        self.assertEqual(result["metrics"]["missing_predictions"], 1)


def test_macro_f1_includes_all_declared_classes():
    references = [{"id": "one", "reference": {"target": "A"}}]
    predictions = [{"id": "one", "probs": {"A": 1, "B": 0, "C": 0}}]
    result = score_predictions.score_all(references, predictions, ["A", "B", "C"])
    assert result["metrics"]["macro_f1"] == pytest.approx(1 / 3)
    assert result["metrics"]["balanced_accuracy"] == 1.0


@pytest.mark.parametrize("probabilities", [{"A": 1}, {"A": 1, "B": 0, "other": 1}])
def test_probabilities_must_cover_exact_taxonomy(probabilities):
    with pytest.raises(ValueError, match="probability labels"):
        score_predictions.normalized_probabilities(
            {"id": "one", "probs": probabilities}, ["A", "B"]
        )


def test_normalization_remains_finite_when_weight_sum_would_overflow():
    actual = score_predictions.normalized_probabilities(
        {"id": "one", "probs": {"A": 1e308, "B": 1e308}}, ["A", "B"]
    )
    assert actual == {"A": 0.5, "B": 0.5}


@pytest.mark.parametrize("labels", [["A", "B"], ["B", "A"]])
def test_saved_prediction_preserves_inference_tie_breaking(labels):
    references = [{"id": "one", "reference": {"target": "B"}}]
    predictions = [{"id": "one", "pred": "B", "probs": {"B": 0.5, "A": 0.5}}]

    metrics = score_predictions.score_all(references, predictions, labels)["metrics"]

    assert metrics["accuracy"] == 1.0
    assert metrics["per_class"]["B"]["predicted"] == 1


@pytest.mark.parametrize("predicted", ["B", "unknown", None, ["A"]])
def test_saved_prediction_must_match_a_maximum_probability_class(predicted):
    with pytest.raises(ValueError, match="maximum-probability taxonomy label"):
        score_predictions.score_all(
            [{"id": "one", "reference": {"target": "A"}}],
            [{"id": "one", "pred": predicted, "probs": {"A": 0.9, "B": 0.1}}],
            ["A", "B"],
        )


def test_extra_prediction_ids_are_rejected_even_in_partial_mode():
    with pytest.raises(ValueError, match="do not correspond"):
        score_predictions.score_all(
            [{"id": "one", "reference": {"target": "A"}}],
            [{"id": "other", "probs": {"A": 1, "B": 0}}],
            ["A", "B"],
            allow_partial=True,
        )


@pytest.mark.parametrize("labels", [[], ["A"], ["A", "A"], ["A", ""], ["A", None]])
def test_taxonomy_labels_must_be_unique_nonempty_strings(labels):
    with pytest.raises(ValueError, match="taxonomy"):
        score_predictions.score_all([], [], labels)


def test_cli_report_uses_configurable_reference_attribution_and_verifies_metadata(tmp_path):
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    taxonomy = tmp_path / "taxonomy.json"
    metadata = tmp_path / "run.json"
    output_json = tmp_path / "score.json"
    output_markdown = tmp_path / "score.md"
    references.write_text(json.dumps({"id": "one", "reference": {"target": "A"}}) + "\n")
    predictions.write_text(json.dumps({"id": "one", "probs": {"A": 0.9, "B": 0.1}}) + "\n")
    taxonomy.write_text(json.dumps({"labels": {"A": "first", "B": "second"}}))
    run = {
        "input_sha256": score_predictions.file_sha256(references),
        "predictions_sha256": score_predictions.file_sha256(predictions),
        "input_rows": 1,
        "predicted_rows": 1,
        "evaluator_failures": 0,
    }
    metadata.write_text(json.dumps(run))
    args = [
        "--references",
        str(references),
        "--predictions",
        str(predictions),
        "--taxonomy",
        str(taxonomy),
        "--run-metadata",
        str(metadata),
        "--target-name",
        "Synthetic reference labels",
        "--output-json",
        str(output_json),
        "--output-markdown",
        str(output_markdown),
    ]
    score_predictions.main(args)
    result = json.loads(output_json.read_text())
    report = output_markdown.read_text()
    assert result["target"] == "Synthetic reference labels"
    assert report.startswith("# Decision model evaluation")
    assert "Synthetic reference labels" in report
    run["predictions_sha256"] = "stale"
    metadata.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="predictions_sha256"):
        score_predictions.main(args)


def test_cli_rejects_overwriting_an_input_before_reading_it(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("preserve me")
    with pytest.raises(SystemExit):
        score_predictions.main(
            [
                "--references",
                str(source),
                "--predictions",
                str(tmp_path / "predictions.jsonl"),
                "--taxonomy",
                str(tmp_path / "taxonomy.json"),
                "--output-json",
                str(source),
                "--output-markdown",
                str(tmp_path / "score.md"),
            ]
        )
    assert source.read_text() == "preserve me"


if __name__ == "__main__":
    unittest.main()
