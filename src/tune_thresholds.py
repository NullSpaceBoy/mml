"""Tune per-label F1 thresholds on a checkpoint's saved validation split.

Example: python src/tune_thresholds.py --checkpoint experiments/run/best \
    --train-csv data/train.csv --output thresholds.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

from common import (
    LABELS,
    fingerprint,
    load_inference,
    load_training,
    predict_texts,
    validate_split,
)


def tune_thresholds(targets, probabilities):
    """Return F1-maximizing per-label thresholds, with higher thresholds breaking ties."""
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        targets.shape != probabilities.shape
        or targets.ndim != 2
        or targets.shape[1] != len(LABELS)
    ):
        raise ValueError(
            f"Targets and probabilities must have shape (n, {len(LABELS)})"
        )
    result = {}
    for column, label in enumerate(LABELS):
        truth, scores = targets[:, column], probabilities[:, column]
        known = ~np.isnan(truth) & (truth != -1)
        truth, scores = truth[known], scores[known]
        if not truth.size:
            raise ValueError(
                f"Cannot tune {label}: validation split has no known targets"
            )
        if not np.isin(truth, [0, 1]).all():
            raise ValueError(f"Targets for {label} must be 0, 1, -1, or NaN")
        precision, recall, thresholds = precision_recall_curve(truth, scores)
        if thresholds.size:
            f1 = (
                2
                * precision[:-1]
                * recall[:-1]
                / (precision[:-1] + recall[:-1] + 1e-15)
            )
            best = np.flatnonzero(f1 == f1.max())
            result[label] = float(thresholds[best[-1]])
        else:
            result[label] = 1.0
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trusted local checkpoint directory",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        required=True,
        help="Original training CSV used to create this checkpoint",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination JSON threshold file"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        model, tokenizer, snapshot, device, directory = load_inference(
            args.checkpoint, args.device
        )
        data = load_training(args.train_csv)
        if fingerprint(args.train_csv) != snapshot["fingerprint"]:
            raise ValueError(
                "Training CSV does not match this checkpoint's fingerprint"
            )
        validate_split(data, snapshot["split"])
        validation = data.set_index("id", drop=False).loc[
            snapshot["split"]["validation"]
        ]
        probabilities = predict_texts(
            model,
            tokenizer,
            validation["comment_text"].tolist(),
            snapshot["config"]["max_length"],
            args.batch_size,
            device,
            description="Tuning thresholds",
        )
        thresholds = tune_thresholds(
            validation[LABELS].to_numpy(dtype=np.float64), probabilities
        )
        output = args.output.expanduser()
        if output.exists() or output.is_symlink():
            raise ValueError(f"Threshold file already exists: {output}")
        output.write_text(
            json.dumps(thresholds, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("Per-label F1-maximizing thresholds:")
        for label in LABELS:
            print(f"  {label}: {thresholds[label]:.8g}")
        print(f"Checkpoint: {directory}")
        print(f"Threshold file: {output}")
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
