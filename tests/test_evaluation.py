"""Download-free tests for CSV handling, masked evaluation, and inference output."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
from common import LABELS, load_training, read_csv, split_training


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        f"bert_baseline_{name}", SOURCE / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_script("test")
inference = load_script("run")


class CsvTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_csv(self, name, columns, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def test_fixed_labels_and_text_preservation(self):
        self.assertEqual(
            LABELS,
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        )
        path = self.write_csv(
            "train.csv",
            ["id", "comment_text", *LABELS],
            [
                ["001", "NA", *([0] * 6)],
                ["NA", 'first\nsecond, quoted "text"', *([1] * 6)],
                ["003", "", *([0] * 6)],
            ],
        )
        frame = load_training(path)
        self.assertEqual(frame["id"].tolist(), ["001", "NA", "003"])
        self.assertEqual(
            frame["comment_text"].tolist(), ["NA", 'first\nsecond, quoted "text"', ""]
        )

    def test_ids_are_unique_and_nonempty_and_schema_is_required(self):
        for name, columns, rows in [
            ("duplicate", ["id", "comment_text"], [["01", "a"], ["01", "b"]]),
            ("blank", ["id", "comment_text"], [["", "a"]]),
            ("whitespace", ["id", "comment_text"], [["  ", "a"]]),
            ("missing", ["id"], [["01"]]),
        ]:
            with self.subTest(name=name):
                path = self.write_csv(f"{name}.csv", columns, rows)
                with self.assertRaises(ValueError):
                    read_csv(path, ["id", "comment_text"])

    def test_duplicate_headers_and_overwide_rows_are_rejected(self):
        duplicate = self.write_csv(
            "duplicate_header.csv",
            ["id", "comment_text", "toxic", *LABELS],
            [["row1", "text", 1, *([0] * 6)]],
        )
        with self.assertRaisesRegex(ValueError, "unique column"):
            load_training(duplicate)
        for identifier in ("row1", "0"):
            with self.subTest(identifier=identifier):
                overwide = self.write_csv(
                    "overwide.csv",
                    ["id", "comment_text", *LABELS],
                    [[identifier, "actual comment", *([0] * 7)]],
                )
                with self.assertRaisesRegex(ValueError, "more fields"):
                    load_training(overwide)

    def test_training_rejects_unknown_or_invalid_labels(self):
        for value in ("", -1, 2, "NA", "inf"):
            with self.subTest(value=value):
                path = self.write_csv(
                    "invalid.csv",
                    ["id", "comment_text", *LABELS],
                    [
                        ["one", "text", value, 0, 0, 0, 0, 0],
                        ["two", "valid", *([0] * 6)],
                    ],
                )
                with self.assertRaises(ValueError):
                    load_training(path)

    def test_stratification_is_disjoint_and_reproducible(self):
        path = self.write_csv(
            "train.csv",
            ["id", "comment_text", *LABELS],
            [
                [f"{row:03d}", "text", *[(row >> column) & 1 for column in range(6)]]
                for row in range(64)
            ],
        )
        frame = load_training(path)
        train, validation = split_training(frame, 0.25, 42)
        again_train, again_validation = split_training(frame, 0.25, 42)
        self.assertEqual(list(train), list(again_train))
        self.assertEqual(list(validation), list(again_validation))
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(frame["id"]))
        self.assertTrue(train)
        self.assertTrue(validation)

    def test_join_is_id_aligned_and_missing_labels_are_unknown(self):
        test_path = self.write_csv(
            "test.csv",
            ["id", "comment_text"],
            [
                ["002", "NA"],
                ["001", ""],
                ["003", "line\none"],
            ],
        )
        labels_path = self.write_csv(
            "labels.csv",
            ["id", *LABELS],
            [
                ["001", 1, -1, "", 0, 1, 0],
                ["002", 0, 1, 0, 1, 0, 1],
                ["extra", *([1] * 6)],
            ],
        )
        frame, targets = evaluation.load_evaluation_data(test_path, labels_path)
        self.assertEqual(frame["id"].tolist(), ["002", "001", "003"])
        self.assertEqual(frame["comment_text"].tolist(), ["NA", "", "line\none"])
        np.testing.assert_array_equal(targets[0], [0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(targets[1], [1, np.nan, np.nan, 0, 1, 0])
        self.assertTrue(np.isnan(targets[2]).all())

    def test_labels_path_is_required_and_duplicate_or_invalid_labels_rejected(self):
        test_path = self.write_csv(
            "test.csv", ["id", "comment_text"], [["001", "text"]]
        )
        with self.assertRaises((ValueError, FileNotFoundError)):
            evaluation.load_evaluation_data(test_path, self.root / "missing.csv")
        for rows in (
            [["001", *([0] * 6)], ["001", *([1] * 6)]],
            [["001", 2, 0, 0, 0, 0, 0]],
            [["extra", "NA", 0, 0, 0, 0, 0]],
            [["extra", "inf", 0, 0, 0, 0, 0]],
        ):
            with self.subTest(rows=rows):
                labels_path = self.write_csv("labels.csv", ["id", *LABELS], rows)
                with self.assertRaises(ValueError):
                    evaluation.load_evaluation_data(test_path, labels_path)

    def test_prediction_csv_keeps_order_and_protects_existing_symlink_target(self):
        probabilities = np.array(
            [[0, 0.2, 0.4, 0.6, 0.8, 1], [1, 0.8, 0.6, 0.4, 0.2, 0]]
        )
        checkpoint = self.root / "epoch_0001"
        checkpoint.mkdir()
        selected = self.root / "best"
        selected.symlink_to(checkpoint, target_is_directory=True)
        output = selected / "predictions.csv"
        evaluation.write_predictions(output, ["002", "001"], probabilities)
        frame = read_csv(output, ["id", *LABELS])
        self.assertEqual(frame.columns.tolist(), ["id", *LABELS])
        self.assertEqual(frame["id"].tolist(), ["002", "001"])
        np.testing.assert_allclose(frame[LABELS].to_numpy(dtype=float), probabilities)
        before = output.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            evaluation.write_predictions(
                checkpoint / "predictions.csv", ["other"], probabilities[:1]
            )
        self.assertEqual(output.read_bytes(), before)
        evaluation.write_predictions(
            output, ["other"], probabilities[:1], overwrite=True
        )
        self.assertEqual(read_csv(output, ["id"])["id"].tolist(), ["other"])

    def test_evaluation_cli_predicts_all_rows_using_checkpoint_settings(self):
        test_path = self.write_csv(
            "test.csv", ["id", "comment_text"], [["02", "NA"], ["01", ""]]
        )
        labels_path = self.write_csv(
            "labels.csv", ["id", *LABELS], [["01", *([0] * 6)]]
        )
        checkpoint = self.root / "epoch_0001"
        checkpoint.mkdir()
        best = self.root / "best"
        best.symlink_to(checkpoint, target_is_directory=True)
        snapshot = {"labels": LABELS, "config": {"max_length": 19}}
        probabilities = np.array([[0.8] * 6, [0.2] * 6])
        arguments = [
            "--checkpoint",
            str(best),
            "--test-csv",
            str(test_path),
            "--test-labels",
            str(labels_path),
            "--batch-size",
            "3",
            "--device",
            "cpu",
        ]
        with (
            patch.object(
                evaluation,
                "load_inference",
                return_value=(
                    "model",
                    "tokenizer",
                    snapshot,
                    "cpu",
                    checkpoint,
                ),
            ),
            patch.object(
                evaluation, "predict_texts", return_value=probabilities
            ) as predict,
        ):
            with redirect_stdout(io.StringIO()) as stdout:
                evaluation.main(arguments)
            predict.assert_called_once_with(
                "model", "tokenizer", ["NA", ""], 19, 3, "cpu"
            )
            self.assertIn("1 without labels", stdout.getvalue())
            output = checkpoint / "predictions.csv"
            self.assertEqual(read_csv(output, ["id"])["id"].tolist(), ["02", "01"])
            predict.reset_mock()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                evaluation.main(arguments)
            self.assertEqual(error.exception.code, 2)
            predict.assert_not_called()
            with redirect_stdout(io.StringIO()):
                evaluation.main([*arguments, "--overwrite"])
            predict.assert_called_once()


class MetricTests(unittest.TestCase):
    def test_masked_entries_and_fully_labeled_exact_match(self):
        targets = np.array(
            [
                [1, 0, 1, 0, 1, 0],
                [0, np.nan, -1, np.nan, np.nan, np.nan],
                [np.nan] * 6,
            ]
        )
        probabilities = np.array(
            [
                [0.5, 0.1, 0.9, 0.2, 0.8, 0.3],
                [0.6, 1, 1, 1, 1, 1],
                [1] * 6,
            ]
        )
        metrics = evaluation.compute_metrics(targets, probabilities)
        self.assertEqual(metrics["entries"], {"evaluated": 7, "excluded": 11})
        self.assertEqual(
            metrics["rows"],
            {
                "total": 3,
                "fully_labeled": 1,
                "partially_labeled": 1,
                "without_labels": 1,
                "excluded_from_exact_match": 2,
            },
        )
        self.assertEqual(metrics["exact_match_accuracy"], 1)
        self.assertAlmostEqual(metrics["hamming_loss"], 1 / 7)
        self.assertAlmostEqual(metrics["micro"]["precision"], 3 / 4)
        self.assertEqual(metrics["micro"]["recall"], 1)
        self.assertAlmostEqual(metrics["micro"]["f1"], 6 / 7)
        self.assertEqual(metrics["micro"]["support"], 3)
        self.assertEqual(metrics["per_label"]["toxic"]["excluded"], 1)
        self.assertEqual(metrics["per_label"]["toxic"]["roc_auc"], 0)
        self.assertEqual(metrics["per_label"]["toxic"]["average_precision"], 0.5)
        self.assertEqual(metrics["macro_valid_counts"]["roc_auc"], 1)
        self.assertEqual(metrics["macro_valid_counts"]["precision"], 6)
        expected_bce = -np.log([0.5, 0.9, 0.9, 0.8, 0.8, 0.7, 0.4]).mean()
        self.assertAlmostEqual(metrics["mean_bce"], expected_bce)

    def test_threshold_is_inclusive_and_probability_metrics_do_not_change(self):
        targets = np.array([[0] * 6, [1] * 6])
        probabilities = np.array([[0.2] * 6, [0.5] * 6])
        inclusive = evaluation.compute_metrics(targets, probabilities, 0.5)
        strict = evaluation.compute_metrics(targets, probabilities, 0.6)
        self.assertEqual(inclusive["micro"]["f1"], 1)
        self.assertEqual(strict["micro"]["f1"], 0)
        for key in ("roc_auc", "average_precision"):
            self.assertEqual(inclusive["micro"][key], strict["micro"][key])
            self.assertEqual(inclusive["macro"][key], strict["macro"][key])
        self.assertEqual(inclusive["mean_bce"], strict["mean_bce"])

    def test_undefined_metrics_and_macro_valid_counts(self):
        targets = np.full((2, 6), np.nan)
        targets[:, 0] = [0, 1]
        targets[:, 1] = 0
        targets[:, 2] = 1
        metrics = evaluation.compute_metrics(targets, np.full((2, 6), 0.5))
        self.assertEqual(
            metrics["macro_valid_counts"],
            {
                "precision": 3,
                "recall": 3,
                "f1": 3,
                "roc_auc": 1,
                "average_precision": 2,
            },
        )
        self.assertEqual(metrics["macro"]["roc_auc"], 0.5)
        self.assertEqual(metrics["macro"]["average_precision"], 0.75)
        self.assertIsNone(metrics["exact_match_accuracy"])
        self.assertIsNone(metrics["per_label"]["severe_toxic"]["roc_auc"])
        self.assertEqual(metrics["per_label"]["obscene"]["average_precision"], 1)
        self.assertIsNone(metrics["per_label"]["severe_toxic"]["average_precision"])
        self.assertEqual(metrics["per_label"]["severe_toxic"]["recall"], 0)
        self.assertIsNone(metrics["per_label"]["threat"]["precision"])
        stream = io.StringIO()
        with redirect_stdout(stream):
            evaluation.print_metrics(metrics)
        self.assertIn("N/A", stream.getvalue())
        self.assertIn("1/6 valid labels", stream.getvalue())

    def test_empty_and_all_unknown_populations(self):
        for targets in (np.empty((0, 6)), np.full((2, 6), np.nan), np.full((2, 6), -1)):
            with self.subTest(shape=targets.shape):
                metrics = evaluation.compute_metrics(targets, np.zeros(targets.shape))
                for key in (
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "average_precision",
                ):
                    self.assertIsNone(metrics["micro"][key])
                    self.assertIsNone(metrics["macro"][key])
                    self.assertEqual(metrics["macro_valid_counts"][key], 0)
                for key in ("mean_bce", "exact_match_accuracy", "hamming_loss"):
                    self.assertIsNone(metrics[key])

    def test_saturated_probability_bce_is_finite(self):
        metrics = evaluation.compute_metrics(
            np.array([[0] * 6, [1] * 6]), np.array([[1] * 6, [0] * 6])
        )
        self.assertTrue(np.isfinite(metrics["mean_bce"]))
        self.assertGreater(metrics["mean_bce"], 30)

    def test_invalid_metric_inputs_are_rejected(self):
        truth = np.zeros((2, 6))
        probabilities = np.full((2, 6), 0.5)
        for targets, scores, threshold in (
            (truth[:, :5], probabilities, 0.5),
            (truth, probabilities[:1], 0.5),
            (truth + 2, probabilities, 0.5),
            (truth + np.inf, probabilities, 0.5),
            (truth, probabilities + 1, 0.5),
            (truth, probabilities * np.nan, 0.5),
            (truth, probabilities, -0.1),
            (truth, probabilities, np.nan),
        ):
            with self.subTest(threshold=threshold, shape=targets.shape):
                with self.assertRaises(ValueError):
                    evaluation.compute_metrics(targets, scores, threshold)


class InferenceTests(unittest.TestCase):
    def test_single_input_stdout_is_only_json_and_empty_text_is_supported(self):
        snapshot = {"labels": LABELS, "config": {"max_length": 17}}

        def load(*args):
            print("library diagnostic")
            return "model", "tokenizer", snapshot, "cpu", Path("checkpoint")

        def predict(*args, **kwargs):
            print("prediction diagnostic")
            self.assertEqual(args[2], [""])
            self.assertEqual(args[3:6], (17, 1, "cpu"))
            return np.array([[0, 0.2, 0.4, 0.6, 0.8, 1]])

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(inference, "load_inference", side_effect=load),
            patch.object(inference, "predict_texts", side_effect=predict),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            inference.main(
                ["--checkpoint", "checkpoint", "--text", "", "--device", "cpu"]
            )
        result = json.loads(stdout.getvalue())
        self.assertEqual(list(result), LABELS)
        self.assertEqual(list(result.values()), [0, 0.2, 0.4, 0.6, 0.8, 1])
        self.assertIn("library diagnostic", stderr.getvalue())
        self.assertIn("prediction diagnostic", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
