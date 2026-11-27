"""Build an offline report from one experiment's saved CSV artifacts.

Example: python src/results.py --experiment bert-baseline --output-dir baseline-report
No inference, checkpoint deserialization, or downloads are performed.
"""

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from common import LABELS, read_csv
from test import compute_metrics, load_targets, print_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    "loss_vs_epoch.png",
    "roc_curves.png",
    "precision_recall_curves.png",
    *(f"confusion_matrix_{label}.png" for label in LABELS),
    "metrics.json",
    "metrics.csv",
)


def directory_name(value):
    if (
        not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
    ):
        raise ValueError(
            "Experiment and output names must be single directory names, not paths"
        )
    return value


def load_history(path):
    """Validate loss history independently of ID-bearing evaluation CSVs."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if not header or len(header) != len(set(header)):
            raise ValueError("Loss history header must contain unique column names")
        required = {"epoch", "train_loss", "validation_loss"}
        if not required.issubset(header):
            raise ValueError(
                "Loss history requires epoch, train_loss, validation_loss columns"
            )
        rows = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError(
                    "Malformed loss history row: field count differs from header"
                )
            rows.append(row)
    if not rows:
        raise ValueError("Loss history must be nonempty")
    history = pd.DataFrame(rows, columns=header)
    for column in ("epoch", "train_loss", "validation_loss"):
        values = pd.to_numeric(history[column], errors="raise")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(
                f"Loss history {column} must contain finite nonnegative values"
            )
        history[column] = values
    epochs = history["epoch"]
    if (epochs <= 0).any() or (epochs % 1 != 0).any() or epochs.duplicated().any():
        raise ValueError("Loss history epochs must be unique positive integers")
    history["epoch"] = epochs.astype(np.int64)
    return history.sort_values("epoch").reset_index(drop=True)


def prepare_curves(targets, probabilities, metrics):
    """Prepare only defined curves, reusing the evaluator's AUC/AP values."""
    known = ~np.isnan(targets) & (targets != -1)
    curves = {"roc": [], "pr": [], "roc_skipped": [], "pr_skipped": []}
    populations = [
        (
            label,
            targets[known[:, i], i],
            probabilities[known[:, i], i],
            metrics["per_label"][label],
        )
        for i, label in enumerate(LABELS)
    ]
    populations.append(
        ("micro", targets[known], probabilities[known], metrics["micro"])
    )
    for name, truth, scores, item in populations:
        if item["roc_auc"] is None:
            curves["roc_skipped"].append(
                f"{name}: ROC undefined (requires both classes)"
            )
        else:
            fpr, tpr, _ = roc_curve(truth, scores)
            curves["roc"].append((name, fpr, tpr, item["roc_auc"]))
        if item["average_precision"] is None:
            curves["pr_skipped"].append(f"{name}: AP undefined (no positive targets)")
        else:
            precision, recall, _ = precision_recall_curve(truth, scores)
            curves["pr"].append((name, recall, precision, item["average_precision"]))
    return curves


def confusion_counts(targets, probabilities, threshold):
    thresholds = np.asarray(threshold, dtype=np.float64)
    if thresholds.ndim == 0:
        thresholds = np.full(len(LABELS), thresholds.item())
    counts = {}
    for i, label in enumerate(LABELS):
        known = ~np.isnan(targets[:, i]) & (targets[:, i] != -1)
        truth = targets[known, i].astype(int)
        predicted = (probabilities[known, i] >= thresholds[i]).astype(int)
        counts[label] = np.bincount(2 * truth + predicted, minlength=4).reshape(2, 2)
    return counts


def save_figure(figure, path):
    try:
        figure.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def write_plots(output, experiment, history, epoch, curves, matrices, metrics):
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(history["epoch"], history["train_loss"], "o-", label="Training loss")
    axis.plot(
        history["epoch"], history["validation_loss"], "o-", label="Validation loss"
    )
    axis.axvline(
        epoch, linestyle="--", color="black", label=f"Selected best epoch: {epoch}"
    )
    axis.set(xlabel="Epoch", ylabel="Loss", title=f"{experiment}: loss history")
    axis.grid(alpha=0.3)
    axis.legend()
    save_figure(figure, output / "loss_vs_epoch.png")
    for kind, filename, xlabel, ylabel, title, score_name in (
        (
            "roc",
            "roc_curves.png",
            "False positive rate",
            "True positive rate",
            "ROC curves",
            "AUC",
        ),
        (
            "pr",
            "precision_recall_curves.png",
            "Recall",
            "Precision",
            "Precision-recall curves",
            "AP",
        ),
    ):
        figure, axis = plt.subplots(figsize=(10, 8))
        for name, x, y, score in curves[kind]:
            axis.plot(
                x,
                y,
                linewidth=2 if name == "micro" else 1.3,
                linestyle="--" if name == "micro" else "-",
                label=f"{name} ({score_name}={score:.4f})",
            )
        if kind == "roc":
            axis.plot([0, 1], [0, 1], ":", color="gray", label="Chance")
        axis.set(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{experiment}: {title}",
            xlim=(0, 1),
            ylim=(0, 1.03),
        )
        axis.grid(alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="best", fontsize=9)
        exclusions = curves[f"{kind}_skipped"]
        if exclusions:
            figure.text(
                0.1, 0.01, "Skipped: " + "\n".join(exclusions), fontsize=9, va="bottom"
            )
        figure.tight_layout(rect=(0, 0.04 + 0.025 * len(exclusions), 1, 1))
        save_figure(figure, output / filename)
    for label, matrix in matrices.items():
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
                fontsize=16,
            )
        item = metrics["per_label"][label]
        threshold = metrics["threshold"]
        label_threshold = threshold[label] if isinstance(threshold, dict) else threshold
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            xlabel="Prediction",
            ylabel="Truth",
            title=f"{experiment}: {label}\nThreshold >= {label_threshold:g}; "
            f"evaluated={item['evaluated']}, excluded={item['excluded']}",
        )
        if not item["evaluated"]:
            figure.text(0.5, 0.01, "No known labels", ha="center")
        save_figure(figure, output / f"confusion_matrix_{label}.png")


def generate_report(
    experiment,
    output_dir,
    test_labels=None,
    threshold=0.5,
    overwrite=False,
    threshold_file=None,
):
    directory_name(experiment)
    directory_name(output_dir)
    if threshold_file is not None:
        from test import load_thresholds

        threshold = load_thresholds(threshold_file)
    source = PROJECT_ROOT / "experiments" / experiment / "best"
    if not source.is_dir():
        raise ValueError(f"Missing or broken best checkpoint directory: {source}")
    checkpoint = source.resolve(strict=True)
    if not checkpoint.is_dir() or not re.fullmatch(r"epoch_[0-9]+", checkpoint.name):
        raise ValueError(
            f"Best checkpoint must resolve to an epoch_NNNN directory: {source}"
        )
    epoch = int(checkpoint.name.removeprefix("epoch_"))
    history = load_history(source.parent / "losses.csv")
    selected = history.loc[history["epoch"] == epoch]
    if selected.empty:
        raise ValueError(f"Selected best epoch {epoch} is absent from loss history")
    predictions = read_csv(checkpoint / "predictions.csv", ["id", *LABELS])
    if predictions.empty:
        raise ValueError("Predictions must be nonempty")
    probabilities = predictions[LABELS].to_numpy(dtype=np.float64)
    label_source = (
        Path(test_labels).expanduser()
        if test_labels is not None
        else PROJECT_ROOT / "data" / "test_labels.csv"
    )
    targets = load_targets(predictions["id"], label_source)
    metrics = compute_metrics(targets, probabilities, threshold)
    curves = prepare_curves(targets, probabilities, metrics)
    matrices = confusion_counts(targets, probabilities, threshold)
    metrics.update(
        experiment=experiment,
        checkpoint_source=str(source),
        checkpoint_resolved=str(checkpoint),
        label_source=str(label_source.resolve()),
        selected_epoch=epoch,
        selected_train_loss=float(selected.iloc[0]["train_loss"]),
        selected_validation_loss=float(selected.iloc[0]["validation_loss"]),
    )
    serialized = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    results_root = PROJECT_ROOT / "results"
    output = results_root / output_dir
    # Resolve against the lexical project path: a symlinked results/ must not escape either.
    if not output.resolve().is_relative_to(results_root.absolute()):
        raise ValueError(
            "Output directory must not escape project results/ through symlinks"
        )
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path is not a directory: {output}")
    for name in ARTIFACTS:
        artifact = output / name
        if artifact.is_symlink():
            raise ValueError(f"Refusing artifact symlink: {artifact}")
        if artifact.exists() and (not overwrite or not artifact.is_file()):
            raise ValueError(
                f"Report artifact already exists: {artifact}; use --overwrite to replace files"
            )
    output.mkdir(parents=True, exist_ok=True)
    try:
        write_plots(output, experiment, history, epoch, curves, matrices, metrics)
        (output / "metrics.json").write_text(serialized, encoding="utf-8")
        fields = [
            "label",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
            "evaluated",
            "excluded",
            "support",
            "negatives",
            "predicted_positive",
        ]
        with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for label in [*LABELS, "micro", "macro"]:
                entry = dict(
                    metrics["per_label"][label] if label in LABELS else metrics[label]
                )
                if label == "micro":
                    entry.update(metrics["entries"])
                    entry["negatives"] = sum(
                        item["negatives"] for item in metrics["per_label"].values()
                    )
                    entry["predicted_positive"] = sum(
                        item["predicted_positive"]
                        for item in metrics["per_label"].values()
                    )
                writer.writerow({"label": label, **entry})
    except Exception as exc:
        raise OSError(f"Unable to write complete report to {output}: {exc}") from exc
    print_metrics(metrics)
    print(f"Report directory: {output}")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Single experiment directory name under project-root experiments/",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Single report directory name under project-root results/",
    )
    parser.add_argument(
        "--test-labels",
        type=Path,
        help="Label CSV relative to the working directory; default: project data/test_labels.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold in [0, 1], default: 0.5; not optimized on test data",
    )
    parser.add_argument(
        "--threshold-file",
        type=Path,
        help="Label-keyed JSON thresholds from src/tune_thresholds.py (overrides --threshold)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only known report artifacts; artifact symlinks are always rejected",
    )
    args = parser.parse_args(argv)
    try:
        generate_report(
            args.experiment,
            args.output_dir,
            args.test_labels,
            args.threshold,
            args.overwrite,
            args.threshold_file,
        )
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
