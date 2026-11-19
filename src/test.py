"""Evaluate a trusted local BERT checkpoint without downloading model assets.

Example: python src/test.py --checkpoint experiments/bert-baseline/best
Use --predictions PATH to select the probability CSV and --overwrite to replace it.
Checkpoints contain pickle data: load only artifacts from a trusted source.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from common import LABELS, load_inference, predict_texts, read_csv


def load_evaluation_data(test_path, labels_path):
    """Join labels by string ID, retaining test order and NaN for unknown targets.

    The labels file must exist and contain all six label columns. Missing label
    rows, blank cells, and -1 are unknown, not negative examples.
    """
    test = read_csv(test_path, ["id", "comment_text"])
    targets = load_targets(test["id"], labels_path)
    test["comment_text"] = test["comment_text"].fillna("").astype(str)
    return test, targets


def load_targets(ids, labels_path):
    """Align validated label CSV targets to unique string IDs in input order."""
    labels = read_csv(labels_path, ["id", *LABELS])
    for label in LABELS:
        values = labels[label].fillna("").astype(str).str.strip()
        try:
            numeric = pd.to_numeric(values.replace("", np.nan), errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Label {label!r} must be 0, 1, -1, or blank") from exc
        if not (numeric.isna() | numeric.isin([-1, 0, 1])).all():
            raise ValueError(f"Label {label!r} must be 0, 1, -1, or blank")
        labels[label] = numeric.mask(numeric == -1, np.nan)
    joined = pd.DataFrame({"id": list(ids)}).merge(
        labels[["id", *LABELS]],
        on="id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    return joined[LABELS].to_numpy(dtype=np.float64)


def _binary_metrics(targets, probabilities, threshold):
    """Metrics for one nonempty population; absent populations are undefined."""
    if not targets.size:
        return dict(
            precision=None,
            recall=None,
            f1=None,
            support=0,
            roc_auc=None,
            average_precision=None,
        )
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        probabilities >= threshold,
        average="binary",
        zero_division=0,
    )
    both_classes = np.unique(targets).size == 2
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(targets.sum()),
        "roc_auc": float(roc_auc_score(targets, probabilities))
        if both_classes
        else None,
        "average_precision": (
            float(average_precision_score(targets, probabilities))
            if np.any(targets == 1)
            else None
        ),
    }


def compute_metrics(targets, probabilities, threshold=0.5):
    """Calculate six-label metrics using only known entries (NaN/-1 excluded).

    Undefined results are None. Macro averages exclude undefined label metrics
    and expose their valid-label counts. AUC requires both target classes;
    AP requires at least one positive target (all-positive AP is 1).
    Probability-based metrics do not depend on the decision threshold.
    """
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != len(LABELS):
        raise ValueError(f"Targets must have shape (n, {len(LABELS)})")
    if probabilities.shape != targets.shape:
        raise ValueError("Probabilities must have the same shape as targets")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if not np.all(
        np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)
    ):
        raise ValueError("Probabilities must be finite values between 0 and 1")
    known = ~np.isnan(targets) & (targets != -1)
    if not np.all(np.isin(targets[known], [0, 1])):
        raise ValueError(
            "Known targets must be 0 or 1; unknown targets must be NaN or -1"
        )
    fully_labeled = known.all(axis=1)
    has_labels = known.any(axis=1)
    per_label = {}
    for column, label in enumerate(LABELS):
        selected = known[:, column]
        truth = targets[selected, column]
        scores = probabilities[selected, column]
        per_label[label] = {
            "evaluated": int(selected.sum()),
            "excluded": int((~selected).sum()),
            "negatives": int((truth == 0).sum()),
            "predicted_positive": int((scores >= threshold).sum()),
            **_binary_metrics(truth, scores, threshold),
        }
    macro, macro_valid_counts = {}, {}
    for metric in ("precision", "recall", "f1", "roc_auc", "average_precision"):
        values = [
            entry[metric] for entry in per_label.values() if entry[metric] is not None
        ]
        macro[metric] = float(np.mean(values)) if values else None
        macro_valid_counts[metric] = len(values)
    truth, scores = targets[known], probabilities[known]
    if truth.size:
        # Clip only for log loss, keeping saturated sigmoid outputs finite.
        epsilon = np.finfo(np.float64).eps
        clipped = np.clip(scores, epsilon, 1 - epsilon)
        bce = float(
            -np.mean(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped))
        )
        hamming = float(np.mean((scores >= threshold) != truth))
    else:
        bce = hamming = None
    exact_match = (
        float(
            np.mean(
                np.all(
                    (probabilities[fully_labeled] >= threshold)
                    == targets[fully_labeled],
                    axis=1,
                )
            )
        )
        if fully_labeled.any()
        else None
    )
    return {
        "threshold": float(threshold),
        "rows": {
            "total": len(targets),
            "fully_labeled": int(fully_labeled.sum()),
            "partially_labeled": int((has_labels & ~fully_labeled).sum()),
            "without_labels": int((~has_labels).sum()),
            "excluded_from_exact_match": int((~fully_labeled).sum()),
        },
        "entries": {"evaluated": int(known.sum()), "excluded": int((~known).sum())},
        "per_label": per_label,
        "micro": _binary_metrics(truth, scores, threshold),
        "macro": macro,
        "macro_valid_counts": macro_valid_counts,
        "hamming_loss": hamming,
        "exact_match_accuracy": exact_match,
        "mean_bce": bce,
    }


def write_predictions(path, ids, probabilities, overwrite=False):
    """Write probabilities in input order; exclusive creation is the default."""
    path = Path(path)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(ids), len(LABELS)):
        raise ValueError(f"Predictions must have shape (n, {len(LABELS)})")
    if not np.all(
        np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)
    ):
        raise ValueError("Probabilities must be finite values between 0 and 1")
    output = pd.DataFrame(probabilities, columns=LABELS)
    output.insert(0, "id", list(ids))
    try:
        with path.open(
            "w" if overwrite else "x", encoding="utf-8", newline=""
        ) as stream:
            output.to_csv(stream, index=False)
    except FileExistsError as exc:
        raise ValueError(
            f"Prediction CSV already exists: {path}; use --overwrite to replace it"
        ) from exc


def print_metrics(metrics):
    """Print human-readable populations, per-label metrics, and aggregates."""

    def display(value):
        return "N/A" if value is None else f"{value:.6f}"

    rows = metrics["rows"]
    entries = metrics["entries"]
    print(f"Threshold: {metrics['threshold']:g} (probability >= threshold)")
    print(
        f"Rows: {rows['total']} total; {rows['fully_labeled']} fully labeled; "
        f"{rows['partially_labeled']} partially labeled; {rows['without_labels']} without labels"
    )
    print(f"Exact-match excluded rows: {rows['excluded_from_exact_match']}")
    print(
        f"Label entries: {entries['evaluated']} evaluated; {entries['excluded']} excluded"
    )
    print("\nPer-label metrics (support = known positive targets):")
    print(
        f"{'Label':<15} {'Known':>7} {'Excluded':>8} {'Support':>7} {'Negative':>8} "
        f"{'Pred+':>7} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'AP':>10}"
    )
    for label, item in metrics["per_label"].items():
        print(
            f"{label:<15} {item['evaluated']:>7} {item['excluded']:>8} {item['support']:>7} "
            f"{item['negatives']:>8} {item['predicted_positive']:>7} "
            + " ".join(
                f"{display(item[key]):>10}"
                for key in (
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "average_precision",
                )
            )
        )
    print("\nAggregate metrics:")
    for average in ("micro", "macro"):
        for key in ("precision", "recall", "f1", "roc_auc", "average_precision"):
            count = (
                f" ({metrics['macro_valid_counts'][key]}/{len(LABELS)} valid labels)"
                if average == "macro"
                else ""
            )
            print(f"  {average} {key}: {display(metrics[average][key])}{count}")
    for key in ("hamming_loss", "exact_match_accuracy", "mean_bce"):
        print(f"  {key}: {display(metrics[key])}")
    print("Unknown labels are excluded; N/A indicates an undefined metric population.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trusted local checkpoint directory, including a run's best/ pointer",
    )
    parser.add_argument(
        "--test-csv",
        "--test-data",
        type=Path,
        required=True,
        help="Test CSV with id and comment_text",
    )
    parser.add_argument(
        "--test-labels",
        "--labels-csv",
        type=Path,
        required=True,
        help="Required labels CSV; unmatched rows and blank/-1 targets are unknown",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Probability CSV, default: selected checkpoint directory/predictions.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing an existing prediction CSV",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold in [0, 1], default: 0.5",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Inference batch size, default: 16"
    )
    parser.add_argument(
        "--device", default="auto", help="Torch device, e.g. auto, cpu, cuda, cuda:0"
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        test, targets = load_evaluation_data(args.test_csv, args.test_labels)
        model, tokenizer, snapshot, device, directory = load_inference(
            args.checkpoint, args.device
        )
        output = (
            args.predictions
            if args.predictions is not None
            else directory / "predictions.csv"
        )
        if not args.overwrite and (output.exists() or output.is_symlink()):
            raise ValueError(
                f"Prediction CSV already exists: {output}; use --overwrite to replace it"
            )
        if not output.parent.is_dir():
            raise ValueError(
                f"Prediction CSV parent directory does not exist: {output.parent}"
            )
        probabilities = predict_texts(
            model,
            tokenizer,
            test["comment_text"].tolist(),
            snapshot["config"]["max_length"],
            args.batch_size,
            device,
        )
        metrics = compute_metrics(targets, probabilities, args.threshold)
        write_predictions(output, test["id"].tolist(), probabilities, args.overwrite)
        print_metrics(metrics)
        print(f"Predictions: {output}")
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
