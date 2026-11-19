"""Download-free regression tests for offline experiment result reports."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
from common import LABELS
import common
import test as evaluation
import results
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


ARTIFACTS = {
    "loss_vs_epoch.png",
    "roc_curves.png",
    "precision_recall_curves.png",
    "metrics.json",
    "metrics.csv",
    *(f"confusion_matrix_{label}.png" for label in LABELS),
}


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.experiment = self.root / "experiments" / "demo"
        self.checkpoint = self.experiment / "checkpoints" / "epoch_0002"
        self.checkpoint.mkdir(parents=True)
        self.best = self.experiment / "best"
        self.best.symlink_to(self.checkpoint, target_is_directory=True)
        self.predictions = self.checkpoint / "predictions.csv"
        self.history = self.experiment / "losses.csv"
        self.labels = self.root / "data" / "test_labels.csv"
        self.labels.parent.mkdir()
        self.output = self.root / "results" / "report"
        self.ids = ["002", "001", "003"]
        self.probabilities = np.array(
            [
                [0.2, 0.8, 0.1, 0.9, 0.2, 0.7],
                [0.5, 0.9, 0.8, 0.1, 0.7, 0.3],
                [1, 1, 1, 1, 1, 1],
            ]
        )
        self.targets = np.array(
            [[0, 1, 0, 1, 0, 1], [1, np.nan, np.nan, 0, 1, 0], [np.nan] * 6]
        )
        self.write_predictions()
        self.write_csv(
            self.labels,
            ["id", *LABELS],
            [
                ["001", 1, -1, "", 0, 1, 0],
                ["extra", *([1] * 6)],
                ["002", 0, 1, 0, 1, 0, 1],
            ],
        )
        self.write_history()
        self.root_patch = patch.object(results, "PROJECT_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    @staticmethod
    def write_csv(path, columns, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)

    def write_predictions(self, rows=None, columns=None):
        if rows is None:
            rows = [
                [identifier, *scores]
                for identifier, scores in zip(self.ids, self.probabilities)
            ]
        self.write_csv(self.predictions, columns or ["id", *LABELS], rows)

    def write_history(self, rows=None, columns=None):
        self.write_csv(
            self.history,
            columns or ["epoch", "train_loss", "validation_loss", "best"],
            rows
            if rows is not None
            else [[3, 0.2, 0.6, False], [1, 0.8, 0.9, True], [2, 0.4, 0.3, True]],
        )

    def generate(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return results.generate_report("demo", "report", **kwargs)

    def assert_rejected(self, **kwargs):
        with self.assertRaises((ValueError, OSError)):
            self.generate(**kwargs)
        self.assertFalse(
            self.output.exists(), "Invalid input created a report directory"
        )

    def read_json(self):
        def invalid_constant(value):
            raise AssertionError(f"Non-standard JSON constant: {value}")

        return json.loads(
            (self.output / "metrics.json").read_text(encoding="utf-8"),
            parse_constant=invalid_constant,
        )

    def test_report_matches_existing_metrics_and_selected_checkpoint(self):
        before = {
            path: path.read_bytes()
            for path in (self.predictions, self.labels, self.history)
        }
        figures = plt.get_fignums()
        metrics = self.generate()
        expected = evaluation.compute_metrics(self.targets, self.probabilities)
        exported = self.read_json()
        for key, value in expected.items():
            self.assertEqual(metrics[key], value, key)
            self.assertEqual(exported[key], value, key)
        self.assertEqual(metrics["experiment"], "demo")
        self.assertEqual(Path(metrics["checkpoint_source"]), self.best)
        self.assertEqual(Path(metrics["checkpoint_resolved"]), self.checkpoint)
        self.assertEqual(Path(metrics["label_source"]), self.labels)
        self.assertEqual(metrics["selected_epoch"], 2)
        self.assertAlmostEqual(metrics["selected_train_loss"], 0.4)
        self.assertAlmostEqual(metrics["selected_validation_loss"], 0.3)
        self.assertEqual(metrics["entries"], {"evaluated": 10, "excluded": 8})
        self.assertEqual(metrics["micro"]["f1"], 1)
        self.assertEqual(metrics["exact_match_accuracy"], 1)
        self.assertEqual({path.name for path in self.output.iterdir()}, ARTIFACTS)
        for name in ARTIFACTS:
            if name.endswith(".png"):
                data = (self.output / name).read_bytes()
                self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
                self.assertGreater(len(data), 1000)
        self.assertEqual(plt.get_fignums(), figures)
        self.assertEqual(plt.get_backend().lower(), "agg")
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        with (self.output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        # The row-name heading is a presentation detail; the populations are not.
        name_column = next(key for key in rows[0] if rows[0][key] == LABELS[0])
        by_name = {row[name_column]: row for row in rows}
        self.assertEqual(set(by_name), {*LABELS, "micro", "macro"})
        for label in LABELS:
            for key in ("precision", "recall", "f1", "roc_auc", "average_precision"):
                value = expected["per_label"][label][key]
                if value is None:
                    self.assertEqual(by_name[label][key], "")
                else:
                    self.assertAlmostEqual(float(by_name[label][key]), value)
        for key in ("evaluated", "excluded", "support"):
            self.assertEqual(
                int(by_name["toxic"][key]), expected["per_label"]["toxic"][key]
            )
            self.assertEqual(by_name["macro"][key], "")

    def test_load_targets_preserves_prediction_order_and_string_ids(self):
        targets = evaluation.load_targets(self.ids, self.labels)
        np.testing.assert_array_equal(targets, self.targets)

    def test_history_is_sorted_by_actual_epochs(self):
        self.write_history([[10, 0.2, 0.4, False], [2, 0.5, 0.3, True]])
        frame = results.load_history(self.history)
        self.assertEqual(frame["epoch"].tolist(), [2, 10])
        self.assertEqual(frame["train_loss"].tolist(), [0.5, 0.2])

    def test_confusion_counts_mask_unknowns_and_include_threshold_equality(self):
        counts = results.confusion_counts(self.targets, self.probabilities, 0.5)
        self.assertEqual(set(counts), set(LABELS))
        np.testing.assert_array_equal(counts["toxic"], [[1, 0], [0, 1]])
        np.testing.assert_array_equal(counts["severe_toxic"], [[0, 0], [0, 1]])
        np.testing.assert_array_equal(counts["obscene"], [[1, 0], [0, 0]])
        high = results.confusion_counts(self.targets, self.probabilities, 0.6)
        np.testing.assert_array_equal(high["toxic"], [[1, 0], [1, 0]])
        unknown = results.confusion_counts(np.full((2, 6), -1), np.zeros((2, 6)), 0.5)
        for matrix in unknown.values():
            np.testing.assert_array_equal(matrix, np.zeros((2, 2)))

    def test_curves_use_masked_populations_and_explain_undefined_curves(self):
        targets = np.full((2, 6), np.nan)
        targets[:, 0] = [0, 1]
        targets[:, 1] = 0
        targets[:, 2] = 1
        probabilities = np.full((2, 6), 0.5)
        metrics = evaluation.compute_metrics(targets, probabilities)
        curves = results.prepare_curves(targets, probabilities, metrics)
        roc = {name: (x, y, score) for name, x, y, score in curves["roc"]}
        pr = {name: (x, y, score) for name, x, y, score in curves["pr"]}
        self.assertEqual(set(roc), {"toxic", "micro"})
        self.assertEqual(set(pr), {"toxic", "obscene", "micro"})
        for name, (_, _, score) in roc.items():
            expected = (
                metrics["micro"] if name == "micro" else metrics["per_label"][name]
            )
            self.assertAlmostEqual(score, expected["roc_auc"])
        for name, (x, y, score) in pr.items():
            self.assertEqual(len(x), len(y))
            expected = (
                metrics["micro"] if name == "micro" else metrics["per_label"][name]
            )
            self.assertAlmostEqual(score, expected["average_precision"])
        for label in LABELS[1:]:
            self.assertTrue(any(label in reason for reason in curves["roc_skipped"]))
        for label in [LABELS[1], *LABELS[3:]]:
            self.assertTrue(any(label in reason for reason in curves["pr_skipped"]))

    def test_no_known_labels_still_exports_strict_json_and_closed_figures(self):
        self.write_csv(self.labels, ["id", *LABELS], [["001", -1, "", -1, "", -1, ""]])
        figures = plt.get_fignums()
        annotations = {}
        original_savefig = Figure.savefig

        def capture(figure, path, *args, **kwargs):
            annotations[Path(path).name] = " ".join(
                text.get_text() for text in figure.texts
            )
            return original_savefig(figure, path, *args, **kwargs)

        with patch.object(Figure, "savefig", autospec=True, side_effect=capture):
            metrics = self.generate()
        exported = self.read_json()
        self.assertEqual(metrics["entries"], {"evaluated": 0, "excluded": 18})
        for key in ("mean_bce", "hamming_loss", "exact_match_accuracy"):
            self.assertIsNone(exported[key])
        self.assertTrue(all(value is None for value in exported["macro"].values()))
        unknown = np.full((3, 6), np.nan)
        curves = results.prepare_curves(unknown, self.probabilities, metrics)
        self.assertEqual(curves["roc"], [])
        self.assertEqual(curves["pr"], [])
        for key in ("roc_skipped", "pr_skipped"):
            self.assertTrue(any("micro" in reason.lower() for reason in curves[key]))
        self.assertEqual({path.name for path in self.output.iterdir()}, ARTIFACTS)
        for filename in ("roc_curves.png", "precision_recall_curves.png"):
            self.assertIn("micro", annotations[filename].lower())
            self.assertIn("undefined", annotations[filename].lower())
        for label in LABELS:
            self.assertIn(
                "no known labels", annotations[f"confusion_matrix_{label}.png"].lower()
            )
        self.assertEqual(plt.get_fignums(), figures)

    def test_loss_plot_uses_actual_epochs_and_only_selected_best_marker(self):
        self.write_history(
            [[10, 0.2, 0.6, False], [1, 0.8, 0.9, True], [2, 0.4, 0.3, True]]
        )
        plots = {}

        def capture(figure, path, *args, **kwargs):
            plots[Path(path).name] = figure

        with patch.object(Figure, "savefig", autospec=True, side_effect=capture):
            self.generate()
        axis = plots["loss_vs_epoch.png"].axes[0]
        for line in axis.lines[:2]:
            np.testing.assert_array_equal(line.get_xdata(), [1, 2, 10])
        self.assertTrue(
            any(np.array_equal(line.get_xdata(), [2, 2]) for line in axis.lines[2:])
        )
        self.assertFalse(
            any(np.array_equal(line.get_xdata(), [1, 1]) for line in axis.lines[2:])
        )
        self.assertTrue(axis.get_xlabel())
        self.assertTrue(axis.get_ylabel())
        self.assertIn("demo", axis.get_title())

    def test_invalid_probability_values_are_rejected_before_output(self):
        for value in ("", "oops", "NaN", "inf", "-inf", -0.01, 1.01):
            with self.subTest(value=value):
                self.write_predictions([["001", value, 0, 0, 0, 0, 0]])
                self.assert_rejected()

    def test_invalid_prediction_csv_structure_is_rejected(self):
        cases = [
            (["id", *LABELS], []),
            (["id", *LABELS], [["001", *([0] * 6)], ["001", *([1] * 6)]]),
            (["id", *LABELS], [[" ", *([0] * 6)]]),
            (["id", *LABELS], [["", *([0] * 6)]]),
            (["id", *LABELS[:-1]], [["001", *([0] * 5)]]),
            (["id", *LABELS, LABELS[0]], [["001", *([0] * 7)]]),
            (["id", *LABELS], [["001", *([0] * 7)]]),
        ]
        for columns, rows in cases:
            with self.subTest(columns=columns, rows=rows):
                self.write_predictions(rows, columns)
                self.assert_rejected()
        self.predictions.write_text('id,toxic\n"unterminated', encoding="utf-8")
        self.assert_rejected()

    def test_invalid_label_values_including_extra_truth_rows_are_rejected(self):
        for value in (2, -2, 0.5, "NA", "nan", "inf", "bad"):
            with self.subTest(value=value):
                self.write_csv(
                    self.labels, ["id", *LABELS], [["extra", value, 0, 0, 0, 0, 0]]
                )
                self.assert_rejected()

    def test_invalid_label_csv_structure_is_rejected(self):
        for columns, rows in (
            (["id", *LABELS], [["001", *([0] * 6)], ["001", *([1] * 6)]]),
            (["id", *LABELS], [[" ", *([0] * 6)]]),
            (["id", *LABELS[:-1]], [["001", *([0] * 5)]]),
            (["id", *LABELS, LABELS[0]], [["001", *([0] * 7)]]),
            (["id", *LABELS], [["001", *([0] * 7)]]),
        ):
            with self.subTest(columns=columns, rows=rows):
                self.write_csv(self.labels, columns, rows)
                self.assert_rejected()

    def test_invalid_history_is_rejected(self):
        cases = [
            [],
            [[1, 0.2, 0.3, True], [1, 0.1, 0.2, True]],
            [[2, 0.2, 0.3]],
            [[2, 0.2, 0.3, True, "extra"]],
        ]
        cases.extend([[epoch, 0.2, 0.3, True]] for epoch in (0, -1, 1.5, "bad", "inf"))
        for value in (-0.1, "nan", "inf", "", "bad"):
            cases.append([[2, value, 0.3, True]])
            cases.append([[2, 0.2, value, True]])
        for rows in cases:
            with self.subTest(rows=rows):
                self.write_history(rows)
                with self.assertRaises((ValueError, OSError)):
                    results.load_history(self.history)
                self.assert_rejected()
        for columns in (
            ["epoch", "train_loss"],
            ["epoch", "train_loss", "validation_loss", "epoch"],
        ):
            self.write_history([], columns)
            self.assert_rejected()

    def test_selected_epoch_must_be_in_history(self):
        self.write_history([[1, 0.2, 0.3, True], [3, 0.1, 0.4, False]])
        self.assert_rejected()

    def test_best_must_resolve_to_an_epoch_directory(self):
        self.best.unlink()
        other = self.experiment / "arbitrary-name"
        other.mkdir()
        self.best.symlink_to(other, target_is_directory=True)
        self.assert_rejected()
        self.best.unlink()
        self.best.symlink_to(self.history)
        self.assert_rejected()

    def test_missing_artifacts_and_broken_best_are_rejected(self):
        for path in (self.predictions, self.history, self.labels):
            with self.subTest(path=path):
                data = path.read_bytes()
                path.unlink()
                try:
                    self.assert_rejected()
                finally:
                    path.write_bytes(data)
        self.best.unlink()
        self.assert_rejected()
        self.best.symlink_to(self.experiment / "missing", target_is_directory=True)
        self.assert_rejected()

    def test_invalid_thresholds_and_directory_names_are_rejected(self):
        for threshold in (-0.01, 1.01, np.nan, np.inf, -np.inf):
            with self.subTest(threshold=threshold):
                self.assert_rejected(threshold=threshold)
        for name in (
            "",
            ".",
            "..",
            "../escape",
            "nested/name",
            "nested\\name",
            str(self.root),
        ):
            for argument in ("experiment", "output_dir"):
                with self.subTest(name=name, argument=argument):
                    with self.assertRaises((ValueError, OSError)):
                        if argument == "experiment":
                            results.generate_report(name, "report")
                        else:
                            results.generate_report("demo", name)
                    self.assertFalse(self.output.exists())

    def test_threshold_boundaries_are_accepted(self):
        for threshold in (0, 1):
            with self.subTest(threshold=threshold):
                metrics = self.generate(threshold=threshold, overwrite=True)
                expected = evaluation.compute_metrics(
                    self.targets, self.probabilities, threshold
                )
                self.assertEqual(metrics["micro"], expected["micro"])

    def test_output_symlink_cannot_escape_results(self):
        self.output.parent.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        self.output.symlink_to(outside, target_is_directory=True)
        for overwrite in (False, True):
            with (
                self.subTest(overwrite=overwrite),
                self.assertRaises((ValueError, OSError)),
            ):
                self.generate(overwrite=overwrite)
        self.assertEqual(list(outside.iterdir()), [])

    def test_results_root_symlink_and_nondirectory_output_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.output.parent.symlink_to(outside, target_is_directory=True)
        self.assert_rejected(overwrite=True)
        self.assertEqual(list(outside.iterdir()), [])
        self.output.parent.unlink()
        self.output.parent.mkdir()
        self.output.write_bytes(b"not a directory")
        with self.assertRaises((ValueError, OSError)):
            self.generate(overwrite=True)
        self.assertEqual(self.output.read_bytes(), b"not a directory")

    def test_preflight_checks_every_artifact_without_changing_existing_bytes(self):
        self.output.mkdir(parents=True)
        for name in ARTIFACTS:
            with self.subTest(name=name):
                artifact = self.output / name
                artifact.write_bytes(b"existing report")
                with self.assertRaises((ValueError, OSError)):
                    self.generate()
                self.assertEqual(artifact.read_bytes(), b"existing report")
                self.assertEqual(list(self.output.iterdir()), [artifact])
                artifact.unlink()

    def test_overwrite_does_not_replace_an_artifact_directory(self):
        artifact = self.output / "metrics.json"
        artifact.mkdir(parents=True)
        protected = artifact / "user-file.txt"
        protected.write_bytes(b"preserve directory contents")
        with self.assertRaises((ValueError, OSError)):
            self.generate(overwrite=True)
        self.assertEqual(protected.read_bytes(), b"preserve directory contents")
        self.assertEqual(list(self.output.iterdir()), [artifact])

    def test_overwrite_preserves_unrelated_files_and_rejects_artifact_symlinks(self):
        self.output.mkdir(parents=True)
        unrelated = self.output / "notes.txt"
        unrelated.write_bytes(b"keep me")
        for name in ARTIFACTS:
            (self.output / name).write_bytes(b"stale")
        with redirect_stdout(io.StringIO()):
            results.main(
                ["--experiment", "demo", "--output-dir", "report", "--overwrite"]
            )
        self.assertEqual(unrelated.read_bytes(), b"keep me")
        for name in ARTIFACTS:
            self.assertNotEqual((self.output / name).read_bytes(), b"stale")
        outside = self.root / "protected.txt"
        outside.write_bytes(b"external")
        artifact = self.output / "metrics.json"
        artifact.unlink()
        artifact.symlink_to(outside)
        for overwrite in (False, True):
            with (
                self.subTest(overwrite=overwrite),
                self.assertRaises((ValueError, OSError)),
            ):
                self.generate(overwrite=overwrite)
        self.assertEqual(outside.read_bytes(), b"external")
        artifact.unlink()
        artifact.symlink_to(self.root / "nonexistent")
        with self.assertRaises((ValueError, OSError)):
            self.generate(overwrite=True)
        self.assertFalse((self.root / "nonexistent").exists())

    def test_cli_uses_project_root_but_explicit_labels_use_working_directory(self):
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        explicit_labels = elsewhere / "labels.csv"
        explicit_labels.write_bytes(self.labels.read_bytes())
        self.labels.unlink()
        previous = Path.cwd()
        stdout = io.StringIO()
        try:
            os.chdir(elsewhere)
            with redirect_stdout(stdout):
                results.main(
                    [
                        "--experiment",
                        "demo",
                        "--output-dir",
                        "report",
                        "--test-labels",
                        "labels.csv",
                        "--threshold",
                        "0.6",
                    ]
                )
        finally:
            os.chdir(previous)
        self.assertTrue(self.output.is_dir())
        self.assertFalse((elsewhere / "results").exists())
        metrics = self.read_json()
        self.assertEqual(metrics["threshold"], 0.6)
        self.assertEqual(Path(metrics["label_source"]).resolve(), explicit_labels)
        self.assertIn(str(self.output), stdout.getvalue())
        self.assertIn("Threshold:", stdout.getvalue())

    def test_cli_reports_actionable_errors_and_does_not_claim_success(self):
        cases = [
            [],
            ["--experiment", "demo"],
            ["--experiment", "missing", "--output-dir", "report"],
            ["--experiment", "demo", "--output-dir", "report", "--threshold", "nan"],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                stderr, stdout = io.StringIO(), io.StringIO()
                with (
                    redirect_stderr(stderr),
                    redirect_stdout(stdout),
                    self.assertRaises(SystemExit) as error,
                ):
                    results.main(arguments)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("error:", stderr.getvalue())
                self.assertFalse(self.output.exists())
        figures = plt.get_fignums()
        with patch.object(Figure, "savefig", side_effect=OSError("disk full")):
            with (
                redirect_stderr(io.StringIO()) as stderr,
                redirect_stdout(io.StringIO()) as stdout,
            ):
                with self.assertRaises(SystemExit) as error:
                    results.main(["--experiment", "demo", "--output-dir", "report"])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("disk full", stderr.getvalue())
            self.assertNotIn(str(self.output), stdout.getvalue())
        self.assertEqual(plt.get_fignums(), figures)

    def test_report_never_loads_checkpoints_constructs_models_or_runs_inference(self):
        with ExitStack() as stack:
            guards = []
            for module in (common, evaluation, results):
                for name in ("load_inference", "predict_texts", "load_checkpoint"):
                    if hasattr(module, name):
                        guards.append(
                            stack.enter_context(
                                patch.object(
                                    module,
                                    name,
                                    side_effect=AssertionError(
                                        "report invoked ML code"
                                    ),
                                )
                            )
                        )
            guards.append(
                stack.enter_context(
                    patch("torch.load", side_effect=AssertionError("checkpoint loaded"))
                )
            )
            for target in (
                "transformers.AutoModel.from_pretrained",
                "transformers.AutoModelForSequenceClassification.from_pretrained",
                "transformers.AutoModelForSequenceClassification.from_config",
                "transformers.AutoConfig.from_pretrained",
                "transformers.AutoTokenizer.from_pretrained",
            ):
                guards.append(
                    stack.enter_context(
                        patch(target, side_effect=AssertionError("model assets loaded"))
                    )
                )
            self.generate()
            for guard in guards:
                guard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
