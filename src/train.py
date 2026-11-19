#!/usr/bin/env python3
r"""Fine-tune a six-label BERT or DistilBERT baseline (single device, full precision).

Examples:
    python train.py --output-dir ../runs/first
    python train.py --resume ../runs/first/checkpoints/epoch_0002 \
        --output-dir ../runs/branch --epochs 10
    python train.py --model bert-base-uncased --weighted \
        --output-dir ../runs/bert-weighted
    python train.py --model distilbert-base-uncased --weighted \
        --output-dir ../runs/distilbert-weighted

Weighted runs automatically estimate per-label negative/positive ratios from
the training split only and use the same weighted BCE for validation, scheduling,
early stopping, and best selection. Inference and test metrics are unchanged.

Epoch targets are absolute. Resume loads trusted local pickle artifacts only.
Every immutable checkpoint retains AdamW state and a full historical best:
expect several GB per BERT epoch, with no automatic pruning. Device changes
are allowed on resume but need not give bitwise-identical results.
"""

import argparse
import copy
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (
    LABELS,
    TextDataset,
    build_model,
    capture_rng,
    choose_device,
    collator,
    fingerprint,
    load_checkpoint,
    load_training,
    package_versions,
    point_best,
    restore_rng,
    save_checkpoint,
    seed_everything,
    split_training,
    validate_split,
    write_history,
)

DEFAULTS = {
    "model": "bert-base-uncased",
    "batch_size": 16,
    "max_length": 256,
    "seed": 42,
    "validation_fraction": 0.1,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "lr_factor": 0.5,
    "lr_patience": 2,
    "min_lr": 1e-7,
    "early_stop_patience": 4,
    "weighted": False,
}


def parser():
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    result.add_argument("--train-csv", type=Path, required=True)
    result.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Must not exist, including for resume",
    )
    result.add_argument(
        "--resume", type=Path, help="Trusted local epoch directory or best symlink"
    )
    result.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Absolute total epoch target (default: 10)",
    )
    result.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    result.add_argument(
        "--weighted",
        action="store_true",
        default=None,
        help="Estimate per-label positive weights from training rows only; use weighted "
        "BCE for training and validation (default: False; inherited on resume)",
    )
    for name, default in DEFAULTS.items():
        if name == "weighted":
            continue
        help_text = f"Default: {default}; inherited on resume"
        if name == "model":
            help_text += "; local pretrained directory or Hub ID (fresh runs only)"
        if name == "lr_patience":
            help_text += "; reduce on this consecutive non-improving epoch"
        result.add_argument(
            "--" + name.replace("_", "-"),
            type=type(default),
            default=None,
            help=help_text,
        )
    return result


def effective_config(args, saved=None):
    config = dict(DEFAULTS if saved is None else saved)
    config.setdefault("weighted", False)
    for key in DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            if saved is not None and value != config[key]:
                raise ValueError(f"Cannot override --{key.replace('_', '-')} on resume")
            config[key] = value
    if not config["model"].strip():
        raise ValueError("model must be a nonempty local path or Hub identifier")
    for key in ("batch_size", "max_length", "lr_patience", "early_stop_patience"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["max_length"] < 2:
        raise ValueError("max_length must leave room for special tokens (at least 2)")
    if not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    for key in (
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "min_lr",
        "validation_fraction",
        "lr_factor",
    ):
        if not math.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0 < config["lr_factor"] < 1:
        raise ValueError("lr_factor must be between 0 and 1")
    if config["learning_rate"] <= 0 or config["gradient_clip"] <= 0:
        raise ValueError("learning_rate and gradient_clip must be positive")
    if (
        not 0 <= config["min_lr"] <= config["learning_rate"]
        or config["weight_decay"] < 0
    ):
        raise ValueError("Require 0 <= min_lr <= learning_rate and weight_decay >= 0")
    return config


def cpu_copy(value):
    """Freeze state without retaining mutable live model/optimizer tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def optimizer_to(optimizer, device):
    # AdamW's scalar step stays on CPU with the default non-capturable optimizer.
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device)


def weighted_criterion(frame, train_ids, device):
    """Estimate positive weights exclusively from the validated training split."""
    targets = torch.tensor(
        frame.set_index("id").loc[train_ids, LABELS].to_numpy(),
        dtype=torch.float64,
    )
    positives = targets.sum(dim=0)
    negatives = len(train_ids) - positives
    invalid = (positives == 0) | (negatives == 0)
    if invalid.any().item():
        labels = ", ".join(
            f"{label} (zero {'positives' if positives[i] == 0 else 'negatives'})"
            for i, label in enumerate(LABELS)
            if invalid[i]
        )
        raise ValueError(
            f"Cannot use --weighted: training labels require both classes: {labels}. "
            "Provide training data/split with positives and negatives for every label."
        )
    weights = (negatives / positives).to(device=device, dtype=torch.float32)
    if not (torch.isfinite(weights) & (weights > 0)).all().item():
        raise ValueError(
            "Cannot use --weighted: positive weights must be finite and positive"
        )
    return torch.nn.BCEWithLogitsLoss(pos_weight=weights)


def epoch_loss(
    model, loader, device, optimizer=None, gradient_clip=1.0, criterion=None
):
    training = optimizer is not None
    model.train(training)
    total, count, steps = 0.0, 0, 0
    progress = tqdm(loader, desc="Train" if training else "Validation", file=sys.stderr)
    with torch.enable_grad() if training else torch.inference_mode():
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            if criterion is None:
                loss = model(**batch).loss
            else:
                inputs = {key: value for key, value in batch.items() if key != "labels"}
                loss = criterion(model(**inputs).logits, batch["labels"].float())
            if not torch.isfinite(loss).item():
                raise ValueError(
                    "Non-finite training/validation loss; no checkpoint saved"
                )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                steps += 1
            size = batch["labels"].shape[0]
            total += loss.item() * size
            count += size
            progress.set_postfix(loss=f"{total / count:.6f}")
    if count == 0:
        raise ValueError("Training and validation populations must both be nonempty")
    return total / count, steps


def update_validation(scheduler, loss, best_loss, bad_epochs):
    if not math.isfinite(loss):
        raise ValueError("Non-finite validation loss")
    improved = loss < best_loss
    scheduler.step(loss)
    return improved, loss if improved else best_loss, 0 if improved else bad_epochs + 1


def train(args):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError(f"Output directory must be fresh: {args.output_dir}")
    device = choose_device(args.device)
    frame = load_training(args.train_csv)
    digest = fingerprint(args.train_csv)
    payload, directory = load_checkpoint(args.resume) if args.resume else (None, None)
    if payload:
        # Normalize both snapshots in memory, preserving legacy best/current equality.
        for name in ("current", "best"):
            payload[name]["config"].setdefault("weighted", False)
    saved = payload["current"] if payload else None
    config = effective_config(args, saved["config"] if saved else None)
    seed_everything(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    if saved:
        if saved["fingerprint"] != digest:
            raise ValueError("Training CSV fingerprint differs from checkpoint")
        split = saved["split"]
        validate_split(frame, split)
        if saved["bad_epochs"] >= config["early_stop_patience"]:
            raise ValueError(
                "Checkpoint has exhausted early stopping; refusing to reset its counter"
            )
        if args.epochs <= saved["epoch"]:
            raise ValueError("Epoch target must exceed completed checkpoint epoch")
    else:
        train_ids, val_ids = split_training(
            frame, config["validation_fraction"], config["seed"]
        )
        split = {"train": train_ids, "validation": val_ids}
        validate_split(frame, split)
    criterion = (
        weighted_criterion(frame, split["train"], device)
        if config["weighted"]
        else None
    )
    if criterion is not None:
        mapping = dict(zip(LABELS, criterion.pos_weight.detach().cpu().tolist()))
        print(
            f"Loss: weighted BCEWithLogitsLoss; pos_weight: {mapping}", file=sys.stderr
        )
    if saved:
        model = build_model(saved, device)
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["model"])
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model"],
            num_labels=len(LABELS),
            problem_type="multi_label_classification",
            id2label=dict(enumerate(LABELS)),
            label2id={label: i for i, label in enumerate(LABELS)},
        ).to(device)
    if config["max_length"] > getattr(
        model.config, "max_position_embeddings", float("inf")
    ):
        raise ValueError("max_length exceeds model position embeddings")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["lr_factor"],
        patience=config["lr_patience"] - 1,
        threshold=0,
        threshold_mode="abs",
        min_lr=config["min_lr"],
        eps=0,
    )
    completed, global_step, best_loss, best_epoch, bad_epochs = 0, 0, math.inf, 0, 0
    history, best = [], None
    if saved:
        # Loading CPU AdamW state can alias tensors in the historical best.
        optimizer.load_state_dict(cpu_copy(saved["optimizer_state"]))
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(copy.deepcopy(saved["scheduler_state"]))
        completed, global_step = saved["epoch"], saved["global_step"]
        best_loss, best_epoch, bad_epochs = (
            saved["best_loss"],
            saved["best_epoch"],
            saved["bad_epochs"],
        )
        history, best = copy.deepcopy(saved["history"]), payload["best"]
    indexed = frame.set_index("id", drop=False)
    loaders = []
    for key in ("train", "validation"):
        subset = indexed.loc[split[key]]
        dataset = TextDataset(
            subset["comment_text"].tolist(), subset[LABELS].to_numpy()
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=key == "train",
                num_workers=0,
                generator=generator if key == "train" else None,
                collate_fn=collator(tokenizer, config["max_length"]),
            )
        )
    args.output_dir.mkdir(parents=True)
    if saved:
        inherited = save_checkpoint(
            args.output_dir,
            f"inherited_best_epoch_{best['epoch']:04d}",
            tokenizer,
            best,
            best,
        )
        point_best(args.output_dir, inherited)
        write_history(args.output_dir, history)
        restore_rng(saved["rng"], generator)
    versions = package_versions()
    print(
        f"Device: {device}; train: {len(split['train'])}; validation: {len(split['validation'])}",
        file=sys.stderr,
    )
    for epoch in range(completed + 1, args.epochs + 1):
        lr_used = optimizer.param_groups[0]["lr"]
        train_loss, steps = epoch_loss(
            model, loaders[0], device, optimizer, config["gradient_clip"], criterion
        )
        val_loss, _ = epoch_loss(model, loaders[1], device, criterion=criterion)
        global_step += steps
        improved, best_loss, bad_epochs = update_validation(
            scheduler, val_loss, best_loss, bad_epochs
        )
        if improved:
            best_epoch = epoch
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "lr_used": lr_used,
                "next_lr": optimizer.param_groups[0]["lr"],
                "global_step": global_step,
                "best": improved,
            }
        )
        current = {
            "config": config,
            "labels": list(LABELS),
            "model_config": model.config.to_dict(),
            "model_state": cpu_copy(model.state_dict()),
            "optimizer_state": cpu_copy(optimizer.state_dict()),
            "scheduler_state": cpu_copy(scheduler.state_dict()),
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "split": split,
            "fingerprint": digest,
            "history": copy.deepcopy(history),
            "rng": capture_rng(generator),
            "versions": versions,
            "execution": {
                "device": str(device),
                "epoch_target": args.epochs,
                "output_dir": str(args.output_dir.resolve()),
                "train_csv": str(args.train_csv.resolve()),
                "num_workers": 0,
                "precision": "float32",
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "torch_num_threads": torch.get_num_threads(),
            },
        }
        if improved:
            best = current
        checkpoint = save_checkpoint(
            args.output_dir, f"epoch_{epoch:04d}", tokenizer, current, best
        )
        if improved:
            point_best(args.output_dir, checkpoint)
        write_history(args.output_dir, history)
        print(
            f"Epoch {epoch}: train={train_loss:.6f} validation={val_loss:.6f} "
            f"lr={lr_used:g} next_lr={optimizer.param_groups[0]['lr']:g} best={improved}",
            file=sys.stderr,
        )
        if bad_epochs >= config["early_stop_patience"]:
            print(
                f"Early stopping after epoch {epoch}; stopping epoch checkpoint committed",
                file=sys.stderr,
            )
            break
    return history


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        train(args)
    except torch.cuda.OutOfMemoryError:
        cli.exit(1, "CUDA out of memory: reduce --batch-size and/or --max-length.\n")
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        cli.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
