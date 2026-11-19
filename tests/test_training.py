"""Offline training integration tests, using temporary tiny transformer assets.

Run with unittest discovery from this directory or the repository root. No test
uses the real dataset, a pretrained Hub model, or a network connection.
"""

import contextlib
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
import common
import train


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="bert-training-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        cls.addClassCleanup(torch.set_num_threads, threads)
        cls.addClassCleanup(torch.use_deterministic_algorithms, deterministic)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(123)
        cls.model_path = cls.root / "tiny-bert"
        cls.model_path.mkdir()
        vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        vocabulary += ["hello", "world", "na", "first", "second", "line"]
        vocabulary += [f"sample{i}" for i in range(24)]
        vocab_path = cls.model_path / "vocab.txt"
        vocab_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
        tokenizer.save_pretrained(cls.model_path)
        config = BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            intermediate_size=20,
            max_position_embeddings=32,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            num_labels=len(common.LABELS),
            problem_type="multi_label_classification",
            id2label=dict(enumerate(common.LABELS)),
            label2id={label: i for i, label in enumerate(common.LABELS)},
        )
        BertForSequenceClassification(config).save_pretrained(cls.model_path)
        rows = []
        for index in range(24):
            text = f"sample{index} hello" + " world" * (index % 3)
            if index == 0:
                text = "first line\nsecond line"
            elif index == 1:
                text = ""
            elif index == 2:
                text = "NA"
            rows.append(
                [f"{index:03d}", text]
                + [(index >> (column % 4)) % 2 for column in range(6)]
            )
        cls.train_csv = cls.root / "train.csv"
        with cls.train_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "comment_text", *common.LABELS])
            writer.writerows(rows)
        cls.test_csv = cls.root / "test.csv"
        cls.test_rows = [["009", "hello world"], ["001", ""], ["003", "NA"]]
        with cls.test_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "comment_text"])
            writer.writerows(cls.test_rows)
        cls.labels_csv = cls.root / "test_labels.csv"
        with cls.labels_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", *common.LABELS])
            writer.writerows([["001", *([1] * 6)], ["009", *([0] * 6)]])
        cls.full = cls.root / "full"
        cls.initial = cls.root / "initial"
        cls.resumed = cls.root / "resumed"
        cls.orders = {}
        cls.run_training(cls.full, 3, record="full")
        cls.run_training(cls.initial, 1, record="initial")
        cls.run_training(
            cls.resumed, 3, resume=cls.epoch(cls.initial, 1), record="resumed"
        )

    @staticmethod
    def epoch(output, number):
        return output / "checkpoints" / f"epoch_{number:04d}"

    @classmethod
    def arguments(cls, output, epochs, resume=None, device="cpu", extra=()):
        arguments = [
            "--train-csv",
            str(cls.train_csv),
            "--output-dir",
            str(output),
            "--epochs",
            str(epochs),
            "--device",
            device,
        ]
        if resume is not None:
            arguments += ["--resume", str(resume)]
        else:
            arguments += [
                "--model",
                str(cls.model_path),
                "--batch-size",
                "5",
                "--max-length",
                "16",
                "--validation-fraction",
                "0.25",
                "--seed",
                "321",
                "--learning-rate",
                "0.003",
                "--early-stop-patience",
                "10",
            ]
        return train.parser().parse_args([*arguments, *extra])

    @classmethod
    def run_training(
        cls,
        output,
        epochs,
        resume=None,
        device="cpu",
        extra=(),
        record=None,
        validation=None,
        inspect_optimizer=None,
        inspect_criterion=None,
    ):
        original = train.epoch_loss
        losses = iter(validation) if validation is not None else None
        observed = []

        def epoch_loss(
            model,
            loader,
            selected_device,
            optimizer=None,
            gradient_clip=1.0,
            criterion=None,
        ):
            if inspect_criterion is not None:
                inspect_criterion(criterion, optimizer, selected_device)
            if optimizer is None:
                if losses is not None:
                    model.eval()
                    return next(losses), 0
                return original(model, loader, selected_device, criterion=criterion)
            if inspect_optimizer is not None:
                inspect_optimizer(model, optimizer, selected_device)
            batches = []

            def capture(module, positional, keywords):
                batches.append(keywords["input_ids"].detach().cpu().tolist())

            handle = model.register_forward_pre_hook(capture, with_kwargs=True)
            try:
                result = original(
                    model, loader, selected_device, optimizer, gradient_clip, criterion
                )
            finally:
                handle.remove()
            observed.append(batches)
            return result

        with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with mock.patch.object(train, "epoch_loss", side_effect=epoch_loss):
                history = train.train(
                    cls.arguments(output, epochs, resume, device, extra)
                )
        if record is not None:
            cls.orders[record] = observed
        return history

    @staticmethod
    def payload(path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return common.load_checkpoint(path)[0]

    def assert_nested_equal(self, actual, expected, location="snapshot"):
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor, location)
            self.assertEqual(actual.dtype, expected.dtype, location)
            self.assertEqual(actual.device, expected.device, location)
            self.assertTrue(torch.equal(actual, expected), location)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected, err_msg=location)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), location)
            for key in expected:
                self.assert_nested_equal(
                    actual[key], expected[key], f"{location}.{key}"
                )
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected), location)
            self.assertEqual(len(actual), len(expected), location)
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.assert_nested_equal(left, right, f"{location}[{index}]")
        else:
            self.assertEqual(actual, expected, location)

    def assert_training_equal(self, actual, expected):
        # Branch provenance differs, but every resumable state must match.
        left, right = copy.deepcopy(actual), copy.deepcopy(expected)
        for payload in (left, right):
            for name in ("current", "best"):
                execution = payload[name].get("execution", {})
                execution.pop("output_dir", None)
                execution.pop("epoch_target", None)
        self.assert_nested_equal(left, right)

    def test_cpu_resume_is_exact_including_next_batches_and_every_snapshot_field(self):
        self.assertEqual(self.orders["full"][:1], self.orders["initial"])
        self.assertEqual(self.orders["full"][1:], self.orders["resumed"])
        for epoch in (2, 3):
            with self.subTest(epoch=epoch):
                expected = self.payload(self.epoch(self.full, epoch))
                actual = self.payload(self.epoch(self.resumed, epoch))
                self.assert_training_equal(actual, expected)
        snapshot = self.payload(self.epoch(self.resumed, 3))["current"]
        self.assertEqual(snapshot["epoch"], 3)
        self.assertEqual(len(snapshot["history"]), 3)
        self.assertEqual(snapshot["labels"], common.LABELS)
        self.assertTrue(snapshot["optimizer_state"]["state"])
        self.assertEqual(
            set(snapshot["rng"]), {"python", "numpy", "cpu", "cuda", "generator"}
        )
        self.assertEqual(snapshot["global_step"], len(self.orders["full"][0]) * 3)
        self.assertEqual(
            (self.full / "losses.csv").read_bytes(),
            (self.resumed / "losses.csv").read_bytes(),
        )

    def test_rng_round_trip_restores_all_sources_and_next_shuffle(self):
        saved = self.payload(self.epoch(self.initial, 1))["current"]["rng"]
        generator = torch.Generator()

        def draw():
            common.restore_rng(saved, generator)
            return {
                "python": random.random(),
                "numpy": np.random.rand(5),
                "cpu": torch.rand(5),
                "shuffle": torch.randperm(24, generator=generator),
                "cuda": [
                    torch.rand(5, device=f"cuda:{i}").cpu()
                    for i in range(torch.cuda.device_count())
                ],
            }

        self.assert_nested_equal(draw(), draw())

    def test_weighted_flags_and_effective_config_including_legacy_resume(self):
        required = ["--train-csv", "train.csv", "--output-dir", "output"]
        omitted = train.parser().parse_args(required)
        enabled = train.parser().parse_args([*required, "--weighted"])
        self.assertIsNone(omitted.weighted)
        self.assertIs(enabled.weighted, True)
        self.assertIs(train.DEFAULTS["weighted"], False)
        fresh = train.effective_config(omitted)
        weighted = train.effective_config(enabled)
        self.assertIs(fresh["weighted"], False)
        self.assertIs(weighted["weighted"], True)
        self.assertEqual(train.effective_config(omitted, weighted), weighted)
        self.assertEqual(train.effective_config(enabled, weighted), weighted)
        legacy = {key: value for key, value in fresh.items() if key != "weighted"}
        for saved in (fresh, legacy):
            with self.subTest(legacy="weighted" not in saved):
                before = copy.deepcopy(saved)
                self.assertEqual(train.effective_config(omitted, saved), fresh)
                with self.assertRaisesRegex(ValueError, "Cannot override --weighted"):
                    train.effective_config(enabled, saved)
                self.assertEqual(saved, before)

    def test_weighted_ratios_are_training_only_in_label_order(self):
        # Deliberately non-positional IDs and an uneven positive count per label.
        targets = np.array(
            [[int(row < count) for count in (1, 2, 3, 4, 5, 2)] for row in range(6)],
            dtype=np.float32,
        )
        frame = pd.DataFrame(targets, columns=common.LABELS)
        frame.insert(0, "id", ["f", "a", "e", "b", "d", "c"])
        validation = frame.copy()
        validation["id"] = "val-" + validation["id"]
        frame = pd.concat([validation, frame], ignore_index=True)
        train_ids = ["c", "d", "b", "e", "a", "f"]
        criterion = train.weighted_criterion(frame, train_ids, torch.device("cpu"))
        expected = torch.tensor([5, 2, 1, 0.5, 0.2, 2], dtype=torch.float32)
        self.assertIsInstance(criterion, torch.nn.BCEWithLogitsLoss)
        self.assertEqual(criterion.reduction, "mean")
        self.assertEqual(criterion.pos_weight.shape, (6,))
        self.assertEqual(criterion.pos_weight.dtype, torch.float32)
        self.assertEqual(criterion.pos_weight.device, torch.device("cpu"))
        self.assertTrue(torch.isfinite(criterion.pos_weight).all())
        self.assertTrue((criterion.pos_weight > 0).all())
        torch.testing.assert_close(criterion.pos_weight, expected, rtol=0, atol=0)
        frame.loc[frame["id"].str.startswith("val-"), common.LABELS] = 1 - targets
        changed = train.weighted_criterion(frame, train_ids, torch.device("cpu"))
        self.assertTrue(torch.equal(changed.pos_weight, criterion.pos_weight))

    def test_degenerate_weighted_labels_fail_before_model_loading_or_output(self):
        frame = common.load_training(self.train_csv)
        train_ids, val_ids = common.split_training(frame, 0.25, 321)
        for value, label in ((0, common.LABELS[1]), (1, common.LABELS[4])):
            with self.subTest(value=value, label=label):
                changed = frame.copy()
                changed.loc[changed["id"].isin(train_ids), label] = value
                # The opposite validation targets must not rescue this class.
                changed.loc[changed["id"].isin(val_ids), label] = 1 - value
                with self.assertRaisesRegex(ValueError, label):
                    train.weighted_criterion(changed, train_ids, torch.device("cpu"))
                csv_path = self.root / f"degenerate-{value}.csv"
                changed.to_csv(csv_path, index=False)
                output = self.root / f"degenerate-{value}"
                with (
                    mock.patch.object(
                        train, "split_training", return_value=(train_ids, val_ids)
                    ),
                    mock.patch.object(
                        train.AutoTokenizer, "from_pretrained"
                    ) as tokenizer,
                    mock.patch.object(
                        train.AutoModelForSequenceClassification, "from_pretrained"
                    ) as model,
                    self.assertRaisesRegex(ValueError, label),
                ):
                    self.run_training(
                        output, 1, extra=("--weighted", "--train-csv", str(csv_path))
                    )
                tokenizer.assert_not_called()
                model.assert_not_called()
                self.assertFalse(output.exists())

    def test_weighted_epoch_raw_logits_uneven_mean_gradient_and_internal_loss(self):
        logits = torch.linspace(-2, 3, 30).reshape(5, 6)
        targets = (torch.arange(30).reshape(5, 6) % 3 == 0).long()
        weights = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        batches = [
            {"input_ids": torch.arange(start, end), "labels": targets[start:end]}
            for start, end in ((0, 2), (2, 4), (4, 5))
        ]

        class FixedLogits(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(logits.clone())
                self.labels_seen = []

            def forward(self, input_ids, labels=None):
                self.labels_seen.append(labels is not None)
                # Intentionally unrelated to BCE: the unweighted path must use it.
                loss = self.logits[input_ids].sum() * 0 + input_ids.float().mean() + 7
                return SimpleNamespace(logits=self.logits[input_ids], loss=loss)

        model = FixedLogits()
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=weights
        ).item()
        with contextlib.redirect_stderr(io.StringIO()):
            actual, steps = train.epoch_loss(
                model, batches, torch.device("cpu"), criterion=criterion
            )
            self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(steps, 0)
            self.assertEqual(model.labels_seen, [False] * 3)
            self.assertFalse(model.training)
            model.labels_seen.clear()
            internal, steps = train.epoch_loss(model, batches, torch.device("cpu"))
            self.assertEqual(internal, 9.0)
            self.assertEqual(steps, 0)
            self.assertEqual(model.labels_seen, [True] * 3)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            trained, steps = train.epoch_loss(
                model, batches, torch.device("cpu"), optimizer, 100.0, criterion
            )
        self.assertAlmostEqual(trained, expected, places=6)
        self.assertEqual(steps, 3)
        self.assertTrue(model.training)
        expected_logits = logits.clone()
        for batch in batches:
            ids = batch["input_ids"]
            values = logits[ids].clone().requires_grad_()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                values, targets[ids].float(), pos_weight=weights
            )
            expected_logits[ids] -= 0.1 * torch.autograd.grad(loss, values)[0]
        torch.testing.assert_close(model.logits, expected_logits)
        self.assertTrue(torch.isfinite(model.logits.grad).all())
        self.assertFalse(torch.equal(model.logits, logits))

    def test_weighted_offline_models_share_train_validation_criterion_and_infer(self):
        distil = self.root / "tiny-distilbert"
        distil.mkdir()
        DistilBertTokenizerFast(
            vocab_file=str(self.model_path / "vocab.txt")
        ).save_pretrained(distil)
        config = DistilBertConfig(
            vocab_size=len(BertTokenizerFast.from_pretrained(self.model_path)),
            dim=12,
            hidden_dim=20,
            n_layers=1,
            n_heads=3,
            max_position_embeddings=32,
            num_labels=6,
            problem_type="multi_label_classification",
            id2label=dict(enumerate(common.LABELS)),
            label2id={label: i for i, label in enumerate(common.LABELS)},
        )
        DistilBertForSequenceClassification(config).save_pretrained(distil)
        for model_path in (self.model_path, distil):
            with self.subTest(model=model_path.name):
                observed = []

                def inspect(criterion, optimizer, device):
                    self.assertIsInstance(criterion, torch.nn.BCEWithLogitsLoss)
                    observed.append((criterion, optimizer is not None))

                output = self.root / f"weighted-{model_path.name}"
                with mock.patch.dict(
                    os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
                ):
                    history = self.run_training(
                        output,
                        1,
                        extra=("--weighted", "--model", str(model_path)),
                        inspect_criterion=inspect,
                    )
                self.assertEqual([training for _, training in observed], [True, False])
                self.assertIs(observed[0][0], observed[1][0])
                state = self.payload(self.epoch(output, 1))["current"]
                self.assertIs(state["config"]["weighted"], True)
                frame = common.load_training(self.train_csv).set_index("id")
                targets = frame.loc[state["split"]["train"], common.LABELS].to_numpy(
                    dtype=np.float64
                )
                expected = torch.tensor(
                    (len(targets) - targets.sum(0)) / targets.sum(0),
                    dtype=torch.float32,
                )
                self.assertTrue(torch.equal(observed[0][0].pos_weight, expected))
                self.assertTrue(
                    all(
                        math.isfinite(history[0][key])
                        for key in ("train_loss", "validation_loss")
                    )
                )
                model, tokenizer, snapshot, device, _ = common.load_inference(
                    output / "best", "cpu"
                )
                probabilities = common.predict_texts(
                    model,
                    tokenizer,
                    ["hello world", ""],
                    snapshot["config"]["max_length"],
                    2,
                    device,
                )
                self.assertEqual(probabilities.shape, (2, 6))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_weighted_cpu_resume_is_exact_including_batches_and_snapshots(self):
        full, initial, resumed = [
            self.root / f"weighted-exact-{name}"
            for name in ("full", "initial", "resumed")
        ]
        self.run_training(full, 3, extra=("--weighted",), record="weighted-full")
        self.run_training(initial, 1, extra=("--weighted",), record="weighted-initial")
        self.run_training(
            resumed, 3, resume=self.epoch(initial, 1), record="weighted-resumed"
        )
        self.assertEqual(
            self.orders["weighted-full"][:1], self.orders["weighted-initial"]
        )
        self.assertEqual(
            self.orders["weighted-full"][1:], self.orders["weighted-resumed"]
        )
        for epoch in (2, 3):
            self.assert_training_equal(
                self.payload(self.epoch(resumed, epoch)),
                self.payload(self.epoch(full, epoch)),
            )
        self.assertEqual(
            (full / "losses.csv").read_bytes(), (resumed / "losses.csv").read_bytes()
        )

    def test_legacy_both_snapshots_normalized_without_new_best_or_source_changes(self):
        original = self.root / "legacy-original"
        self.run_training(original, 2, validation=[1.0, 2.0])
        payload = self.payload(self.epoch(original, 2))
        self.assertNotEqual(payload["current"]["epoch"], payload["best"]["epoch"])
        for name in ("current", "best"):
            payload[name]["config"].pop("weighted", None)
        tokenizer = BertTokenizerFast.from_pretrained(
            self.model_path, local_files_only=True
        )
        legacy = common.save_checkpoint(
            self.root / "legacy",
            "epoch_0002",
            tokenizer,
            payload["current"],
            payload["best"],
        )
        before = {
            path.name: path.read_bytes() for path in legacy.iterdir() if path.is_file()
        }
        rejected = self.root / "legacy-weighted-rejected"
        with self.assertRaisesRegex(ValueError, "Cannot override --weighted"):
            self.run_training(rejected, 3, resume=legacy, extra=("--weighted",))
        self.assertFalse(rejected.exists())
        observed = []
        output = self.root / "legacy-resumed"
        self.run_training(
            output,
            3,
            resume=legacy,
            validation=[3.0],
            inspect_criterion=lambda criterion, optimizer, device: observed.append(
                criterion
            ),
        )
        self.assertEqual(observed, [None, None])
        resumed = self.payload(self.epoch(output, 3))
        expected_best = copy.deepcopy(payload["best"])
        expected_best["config"]["weighted"] = False
        self.assert_nested_equal(resumed["best"], expected_best)
        self.assertIs(resumed["current"]["config"]["weighted"], False)
        self.assertFalse(resumed["current"]["history"][-1]["best"])
        inherited = self.payload(output / "best")
        self.assert_nested_equal(inherited["current"], expected_best)
        self.assertIs(inherited["current"], inherited["best"])
        self.assertEqual(
            before,
            {
                path.name: path.read_bytes()
                for path in legacy.iterdir()
                if path.is_file()
            },
        )

    def test_scheduler_strict_improvement_second_bad_reduction_and_fourth_bad_stop(
        self,
    ):
        output = self.root / "controlled"
        history = self.run_training(
            output,
            9,
            extra=("--early-stop-patience", "4"),
            validation=[1.0, 1.0, 1.1, 1.2, 1.3],
        )
        self.assertEqual(len(history), 5)
        self.assertEqual(
            [row["best"] for row in history], [True, False, False, False, False]
        )
        rate = history[0]["lr_used"]
        self.assertEqual(
            [row["next_lr"] for row in history],
            [rate, rate, rate / 2, rate / 2, rate / 4],
        )
        self.assertEqual(
            [row["lr_used"] for row in history], [rate, rate, rate, rate / 2, rate / 2]
        )
        stopped = self.payload(self.epoch(output, 5))
        self.assertEqual(stopped["current"]["bad_epochs"], 4)
        self.assertEqual(stopped["current"]["scheduler_state"]["last_epoch"], 5)
        self.assert_nested_equal(stopped["current"]["history"], history)
        self.assertFalse(self.epoch(output, 6).exists())
        persisted = pd.read_csv(output / "losses.csv")
        self.assertEqual(persisted["epoch"].tolist(), [1, 2, 3, 4, 5])
        best = output / "best"
        self.assertTrue(best.is_symlink())
        self.assertFalse(Path(os.readlink(best)).is_absolute())
        self.assertEqual(best.resolve(), self.epoch(output, 1).resolve())
        best_payload = self.payload(best)
        self.assertIs(best_payload["current"], best_payload["best"])
        self.assert_nested_equal(stopped["best"], best_payload["current"])
        self.assertEqual(best_payload["current"]["bad_epochs"], 0)
        self.assertEqual(
            best_payload["current"]["optimizer_state"]["param_groups"][0]["lr"], rate
        )
        rejected = self.root / "exhausted"
        with self.assertRaisesRegex(ValueError, "exhausted early stopping"):
            self.run_training(rejected, 6, resume=self.epoch(output, 5))
        self.assertFalse(rejected.exists())

        # Branch from an older non-best epoch, not from the final checkpoint.
        branch = self.root / "historical-branch"
        self.run_training(branch, 4, resume=self.epoch(output, 3), validation=[1.05])
        inherited = branch / "checkpoints" / "inherited_best_epoch_0001"
        self.assertTrue(inherited.is_dir())
        self.assertEqual((branch / "best").resolve(), inherited.resolve())
        inherited_payload = self.payload(inherited)
        self.assertIs(inherited_payload["current"], inherited_payload["best"])
        self.assert_nested_equal(inherited_payload["current"], best_payload["current"])
        branch_state = self.payload(self.epoch(branch, 4))
        self.assert_nested_equal(branch_state["best"], best_payload["current"])
        self.assertEqual(branch_state["current"]["bad_epochs"], 3)
        self.assertEqual(len(branch_state["current"]["history"]), 4)
        self.assertFalse(self.epoch(branch, 5).exists())

        alias_branch = self.root / "best-alias-branch"
        self.run_training(
            alias_branch, 2, resume=self.epoch(output, 1), validation=[1.0]
        )
        alias_payload = self.payload(self.epoch(alias_branch, 2))
        self.assert_nested_equal(alias_payload["best"], best_payload["current"])
        self.assertEqual(alias_payload["current"]["bad_epochs"], 1)

    def test_strict_improvement_resets_counters_and_nonfinite_losses_are_rejected(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=1, factor=0.5, threshold=0, threshold_mode="abs"
        )
        best, bad = math.inf, 0
        for loss, expected in [
            (1.0, (True, 1.0, 0)),
            (1.0, (False, 1.0, 1)),
            (0.999999999, (True, 0.999999999, 0)),
            (1.1, (False, 0.999999999, 1)),
        ]:
            improved, best, bad = train.update_validation(scheduler, loss, best, bad)
            self.assertEqual((improved, best, bad), expected)
        previous = copy.deepcopy(scheduler.state_dict())
        for loss in (float("nan"), float("inf"), -float("inf")):
            with (
                self.subTest(loss=loss),
                self.assertRaisesRegex(ValueError, "Non-finite"),
            ):
                train.update_validation(scheduler, loss, best, bad)
            self.assert_nested_equal(scheduler.state_dict(), previous)
        output = self.root / "nonfinite"
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.run_training(output, 1, validation=[float("nan")])
        self.assertFalse(self.epoch(output, 1).exists())

    def test_resume_rejects_data_settings_split_epoch_and_existing_output_mismatches(
        self,
    ):
        checkpoint = self.epoch(self.initial, 1)
        changed = self.root / "changed.csv"
        changed.write_bytes(
            self.train_csv.read_bytes().replace(b"first line", b"changed line", 1)
        )
        cases = [
            (checkpoint, 2, ("--train-csv", str(changed)), "fingerprint"),
            (checkpoint, 2, ("--batch-size", "7"), "Cannot override"),
            (checkpoint, 2, ("--seed", "42"), "Cannot override"),
            (checkpoint, 1, (), "target must exceed"),
        ]
        corrupt = copy.deepcopy(self.payload(checkpoint))
        for snapshot in (corrupt["current"], corrupt["best"]):
            if snapshot["split"]["train"][-1] != "missing-row":
                snapshot["split"]["train"][-1] = "missing-row"
        tokenizer = BertTokenizerFast.from_pretrained(checkpoint, local_files_only=True)
        invalid_split = common.save_checkpoint(
            self.root / "bad-split",
            "epoch_0001",
            tokenizer,
            corrupt["current"],
            corrupt["best"],
        )
        cases.append((invalid_split, 2, (), "split does not exactly cover"))
        for index, (source, epochs, extra, message) in enumerate(cases):
            output = self.root / f"rejected-{index}"
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.run_training(output, epochs, resume=source, extra=extra)
            self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "must be fresh"):
            self.run_training(self.initial, 2, resume=checkpoint)

    def test_copied_checkpoint_is_self_contained_without_original_model_or_run(self):
        source_model = self.root / "disposable-model"
        shutil.copytree(self.model_path, source_model)
        source_run = self.root / "disposable-run"
        self.run_training(source_run, 1, extra=("--model", str(source_model)))
        copied = self.root / "copied-checkpoint"
        shutil.copytree(self.epoch(source_run, 1), copied)
        shutil.rmtree(source_run)
        shutil.rmtree(source_model)
        output = self.root / "offline-branch"
        with mock.patch.object(
            train.AutoModelForSequenceClassification,
            "from_pretrained",
            side_effect=AssertionError("resume must not load source model"),
        ):
            self.run_training(output, 2, resume=copied)
        loaded = common.load_inference(copied, "cpu")
        model, tokenizer, snapshot, device, _ = loaded
        probabilities = common.predict_texts(
            model,
            tokenizer,
            ["hello world", ""],
            snapshot["config"]["max_length"],
            2,
            device,
        )
        self.assertEqual(probabilities.shape, (2, 6))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertEqual(self.payload(self.epoch(output, 2))["current"]["epoch"], 2)

    def test_interrupted_checkpoint_and_pointer_writes_keep_committed_state(self):
        output = self.root / "atomic-checkpoints"
        payload = self.payload(self.epoch(self.initial, 1))
        tokenizer = BertTokenizerFast.from_pretrained(
            self.epoch(self.initial, 1), local_files_only=True
        )
        committed = common.save_checkpoint(
            output, "epoch_0001", tokenizer, payload["current"], payload["best"]
        )
        common.point_best(output, committed)
        original_bytes = (committed / "checkpoint.pt").read_bytes()

        def partial_write(value, stream):
            stream.write(b"interrupted pickle")
            raise failure("simulated disk failure")

        for failure in (OSError, RuntimeError):
            with self.subTest(failure=failure.__name__):
                with mock.patch.object(common.torch, "save", side_effect=partial_write):
                    with self.assertRaisesRegex(OSError, "simulated disk failure"):
                        common.save_checkpoint(
                            output,
                            "epoch_0002",
                            tokenizer,
                            payload["current"],
                            payload["best"],
                        )
                self.assertFalse(self.epoch(output, 2).exists())
        self.assertFalse(self.epoch(output, 2).exists())
        self.assertEqual(
            sorted(path.name for path in (output / "checkpoints").iterdir()),
            ["epoch_0001"],
        )
        with mock.patch.object(
            common.os, "replace", side_effect=OSError("pointer failure")
        ):
            with self.assertRaisesRegex(OSError, "pointer failure"):
                common.point_best(output, committed)
        self.assertEqual((output / "best").resolve(), committed.resolve())
        self.assertEqual((committed / "checkpoint.pt").read_bytes(), original_bytes)
        self.assert_nested_equal(self.payload(output / "best"), payload)
        self.assertFalse(list(output.glob(".best-*")))
        with self.assertRaises(FileExistsError):
            common.save_checkpoint(
                output, "epoch_0001", tokenizer, payload["current"], payload["best"]
            )

    def test_history_interruption_preserves_csv_and_checkpoint_history_recovers(self):
        output = self.root / "atomic-history"
        history = self.payload(self.epoch(self.initial, 1))["current"]["history"]
        common.write_history(output, history)
        committed_csv = (output / "losses.csv").read_bytes()
        with mock.patch.object(
            common.os, "replace", side_effect=OSError("history failure")
        ):
            with self.assertRaisesRegex(OSError, "history failure"):
                common.write_history(output, history + history)
        self.assertEqual((output / "losses.csv").read_bytes(), committed_csv)
        self.assertFalse(list(output.glob(".losses-*")))

        interrupted = self.root / "interrupted-run"
        write_history = common.write_history
        calls = 0

        def interrupt_epoch_history(directory, rows):
            nonlocal calls
            calls += 1
            if calls == 2:
                with mock.patch.object(
                    common.os, "replace", side_effect=OSError("history failure")
                ):
                    write_history(directory, rows)
            else:
                write_history(directory, rows)

        with mock.patch.object(
            train, "write_history", side_effect=interrupt_epoch_history
        ):
            with self.assertRaisesRegex(OSError, "history failure"):
                self.run_training(interrupted, 2, resume=self.epoch(self.initial, 1))
        checkpoint = self.epoch(interrupted, 2)
        self.assertTrue(checkpoint.is_dir())
        self.assertEqual(len(self.payload(checkpoint)["current"]["history"]), 2)
        self.assertEqual(len(pd.read_csv(interrupted / "losses.csv")), 1)
        recovered = self.root / "history-recovered"
        self.run_training(recovered, 3, resume=checkpoint)
        self.assert_training_equal(
            self.payload(self.epoch(recovered, 3)),
            self.payload(self.epoch(self.full, 3)),
        )
        self.assertEqual(
            (recovered / "losses.csv").read_bytes(),
            (self.full / "losses.csv").read_bytes(),
        )

    def cli(self, script, *arguments, device="cpu"):
        environment = dict(os.environ)
        environment.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        result = subprocess.run(
            [sys.executable, str(SOURCE / script), *map(str, arguments)],
            cwd=self.root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"{script} ({device})\n{result.stdout}\n{result.stderr}",
        )
        return result

    def check_inference_clis(self, checkpoint, device, prefix):
        model, tokenizer, snapshot, selected, _ = common.load_inference(
            checkpoint, device
        )
        texts = [row[1] for row in self.test_rows]
        expected = common.predict_texts(
            model, tokenizer, texts, snapshot["config"]["max_length"], 2, selected
        )
        self.assertTrue(np.isfinite(expected).all())
        predictions = self.root / f"{prefix}-predictions.csv"
        result = self.cli(
            "test.py",
            "--checkpoint",
            checkpoint,
            "--test-csv",
            self.test_csv,
            "--test-labels",
            self.labels_csv,
            "--predictions",
            predictions,
            "--batch-size",
            "2",
            "--device",
            device,
            device=device,
        )
        self.assertIn("mean_bce", result.stdout)
        self.assertIn("3 total", result.stdout)
        frame = pd.read_csv(predictions, dtype={"id": str})
        self.assertEqual(frame.columns.tolist(), ["id", *common.LABELS])
        self.assertEqual(frame["id"].tolist(), [row[0] for row in self.test_rows])
        np.testing.assert_allclose(
            frame[common.LABELS].to_numpy(), expected, rtol=1e-6, atol=1e-7
        )
        for index in (0, 1):
            result = self.cli(
                "run.py",
                "--checkpoint",
                checkpoint,
                "--text",
                texts[index],
                "--device",
                device,
                device=device,
            )
            values = json.loads(result.stdout)
            self.assertEqual(list(values), common.LABELS)
            self.assertTrue(
                all(
                    isinstance(value, float) and 0 <= value <= 1
                    for value in values.values()
                )
            )
            np.testing.assert_allclose(
                list(values.values()), expected[index], rtol=1e-6, atol=1e-7
            )

    def test_all_cpu_clis_are_offline_and_single_text_matches_batch_predictions(self):
        for script in ("train.py", "test.py", "run.py"):
            with self.subTest(script=script):
                self.assertIn("usage:", self.cli(script, "--help").stdout)
        output = self.root / "cli-cpu"
        self.cli(
            "train.py",
            "--train-csv",
            self.train_csv,
            "--output-dir",
            output,
            "--model",
            self.model_path,
            "--epochs",
            "1",
            "--batch-size",
            "5",
            "--max-length",
            "16",
            "--validation-fraction",
            "0.25",
            "--device",
            "cpu",
        )
        state = self.payload(self.epoch(output, 1))["current"]
        self.assertEqual(state["epoch"], 1)
        self.assertTrue(math.isfinite(state["history"][0]["validation_loss"]))
        self.check_inference_clis(output / "best", "cpu", "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_weighted_cuda_criterion_placement_and_resume(self):
        initial = self.root / "weighted-cuda-initial"
        resumed = self.root / "weighted-cuda-resumed"
        observed = []

        def inspect(criterion, optimizer, device):
            self.assertEqual(device.type, "cuda")
            self.assertEqual(criterion.pos_weight.device.type, "cuda")
            self.assertEqual(criterion.pos_weight.dtype, torch.float32)
            self.assertTrue(torch.isfinite(criterion.pos_weight).all())
            observed.append(criterion)

        deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)
        self.run_training(
            initial, 1, device="cuda", extra=("--weighted",), inspect_criterion=inspect
        )
        saved = self.payload(self.epoch(initial, 1))["current"]
        inspected = []

        def inspect_optimizer(model, optimizer, device):
            self.assert_nested_equal(
                train.cpu_copy(optimizer.state_dict()), saved["optimizer_state"]
            )
            self.assertTrue(
                all(parameter.device.type == "cuda" for parameter in model.parameters())
            )
            for states in optimizer.state.values():
                for key, value in states.items():
                    if isinstance(value, torch.Tensor):
                        self.assertEqual(
                            value.device.type, "cpu" if key == "step" else "cuda"
                        )
            inspected.append(True)

        self.run_training(
            resumed,
            2,
            resume=self.epoch(initial, 1),
            device="cuda",
            inspect_criterion=inspect,
            inspect_optimizer=inspect_optimizer,
        )
        self.assertEqual(inspected, [True])
        self.assertEqual(len(observed), 4)
        self.assertIs(observed[0], observed[1])
        self.assertIs(observed[2], observed[3])
        self.assertTrue(torch.equal(observed[0].pos_weight, observed[2].pos_weight))
        state = self.payload(self.epoch(resumed, 2))["current"]
        self.assertIs(state["config"]["weighted"], True)
        self.assertEqual(state["epoch"], 2)
        self.assertGreater(state["global_step"], saved["global_step"])
        self.assertTrue(
            all(
                math.isfinite(row[key])
                for row in state["history"]
                for key in ("train_loss", "validation_loss")
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_training_resume_rng_optimizer_placement_and_inference(self):
        initial, resumed = self.root / "cuda-initial", self.root / "cuda-resumed"
        deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)
        self.run_training(initial, 1, device="cuda")
        saved = self.payload(self.epoch(initial, 1))["current"]
        self.assertEqual(len(saved["rng"]["cuda"]), torch.cuda.device_count())
        generator = torch.Generator()
        common.restore_rng(saved["rng"], generator)
        expected_draw = torch.rand(8, device="cuda").cpu()
        common.restore_rng(saved["rng"], generator)
        self.assertTrue(torch.equal(torch.rand(8, device="cuda").cpu(), expected_draw))
        inspected = []

        def inspect_optimizer(model, optimizer, device):
            self.assertEqual(device.type, "cuda")
            self.assert_nested_equal(
                torch.cuda.get_rng_state_all(), saved["rng"]["cuda"]
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), saved["rng"]["cpu"]))
            self.assertEqual(random.getstate(), saved["rng"]["python"])
            self.assert_nested_equal(np.random.get_state(), saved["rng"]["numpy"])
            restored = train.cpu_copy(optimizer.state_dict())
            self.assert_nested_equal(restored, saved["optimizer_state"])
            self.assertTrue(
                all(parameter.device.type == "cuda" for parameter in model.parameters())
            )
            self.assertTrue(optimizer.state)
            for states in optimizer.state.values():
                for key, value in states.items():
                    if isinstance(value, torch.Tensor):
                        self.assertEqual(
                            value.device.type, "cpu" if key == "step" else "cuda"
                        )
                        self.assertTrue(torch.isfinite(value).all())
            inspected.append(True)

        self.run_training(
            resumed,
            2,
            resume=self.epoch(initial, 1),
            device="cuda",
            inspect_optimizer=inspect_optimizer,
        )
        self.assertEqual(inspected, [True])
        state = self.payload(self.epoch(resumed, 2))["current"]
        self.assertEqual(state["epoch"], 2)
        self.assertEqual(len(state["history"]), 2)
        self.assertGreater(state["global_step"], saved["global_step"])
        self.assertTrue(
            all(
                math.isfinite(row[field])
                for row in state["history"]
                for field in ("train_loss", "validation_loss")
            )
        )
        self.assertFalse(torch.equal(saved["rng"]["cuda"][0], state["rng"]["cuda"][0]))
        self.check_inference_clis(self.epoch(resumed, 2), "cuda", "cuda")


if __name__ == "__main__":
    unittest.main()
