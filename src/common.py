"""Shared data, reproducibility, and trusted-local checkpoint utilities.

Checkpoints contain pickled Python state. Only load artifacts you created or
otherwise trust; local-only loading is not a sandbox for untrusted checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "validation_loss",
    "lr_used",
    "next_lr",
    "global_step",
    "best",
]
_CHECKPOINT_FILE = "checkpoint.pt"
_SNAPSHOT_FIELDS = {
    "config",
    "labels",
    "model_config",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "epoch",
    "global_step",
    "best_loss",
    "best_epoch",
    "bad_epochs",
    "split",
    "fingerprint",
    "history",
    "rng",
    "versions",
}


def _validate_ids(df: pd.DataFrame) -> None:
    if "id" not in df.columns:
        raise ValueError("CSV is missing required column: id")
    if (
        not df["id"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise ValueError("CSV ids must be nonempty strings (blank ids are not allowed)")
    duplicates = df.loc[df["id"].duplicated(), "id"].tolist()
    if duplicates:
        raise ValueError(
            f"CSV ids must be unique; duplicate ids include {duplicates[:5]!r}"
        )


def read_csv(path, required) -> pd.DataFrame:
    """Read a CSV without interpreting literal NA or empty comments as nulls."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist or is not a file: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if not header or len(header) != len(set(header)):
            raise ValueError("CSV header must be nonempty with unique column names")
        df = pd.read_csv(
            path,
            dtype={"id": str, "comment_text": str},
            keep_default_na=False,
            na_filter=False,
            skip_blank_lines=False,
        )
    except OSError as exc:
        raise OSError(f"Unable to read CSV {path}: {exc}") from exc
    except (ValueError, csv.Error, pd.errors.ParserError) as exc:
        raise ValueError(f"Invalid CSV {path}: {exc}") from exc
    # Pandas can consume extra fields as an implicit index instead of failing.
    if not isinstance(df.index, pd.RangeIndex):
        raise ValueError(
            f"Invalid CSV {path}: rows contain more fields than the header"
        )
    missing = sorted(set(required).union({"id"}).difference(df.columns))
    if missing:
        raise ValueError(
            f"CSV {path} is missing required columns: {', '.join(missing)}"
        )
    _validate_ids(df)
    if "comment_text" in df.columns:
        df["comment_text"] = df["comment_text"].fillna("").astype(str)
    return df


def load_training(path) -> pd.DataFrame:
    """Load training rows and validate all six targets as finite binary floats."""
    df = read_csv(path, ["id", "comment_text", *LABELS])
    for label in LABELS:
        try:
            values = pd.to_numeric(df[label], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Training label {label!r} must contain only binary 0/1 values"
            ) from exc
        if not values.isin([0, 1]).all():
            raise ValueError(
                f"Training label {label!r} must contain only finite binary 0/1 values"
            )
        df[label] = values.astype(np.float32)
    return df


def fingerprint(path) -> str:
    """SHA256 of the exact file bytes, including CSV ordering and formatting."""
    path = Path(path).expanduser()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OSError(f"Unable to fingerprint training CSV {path}: {exc}") from exc
    return digest.hexdigest()


def split_training(df: pd.DataFrame, fraction: float, seed: int) -> tuple[list, list]:
    """Iterative multilabel stratification; preserve original row order per split."""
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError(
            "Validation fraction must be finite and strictly between 0 and 1"
        )
    _validate_ids(df)
    if len(df) < 2:
        raise ValueError("At least two training rows are required for stratification")
    targets = df[LABELS].to_numpy(dtype=np.float32)
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Stratification requires finite binary training targets")
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=fraction, random_state=seed
    )
    try:
        train_indices, validation_indices = next(
            splitter.split(np.zeros((len(df), 1)), targets)
        )
    except ValueError as exc:
        raise ValueError(
            f"Cannot split {len(df)} training rows with fraction {fraction}: {exc}"
        ) from exc
    train_ids = df.iloc[np.sort(train_indices)]["id"].tolist()
    validation_ids = df.iloc[np.sort(validation_indices)]["id"].tolist()
    validate_split(df, {"train": train_ids, "validation": validation_ids})
    return train_ids, validation_ids


def _validate_split_lists(split) -> None:
    if not isinstance(split, dict) or set(split) != {"train", "validation"}:
        raise ValueError(
            "Saved split must contain exactly 'train' and 'validation' ID lists"
        )
    for name, ids in split.items():
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Saved {name} split must be a nonempty ID list")
        if not all(isinstance(value, str) and value.strip() for value in ids):
            raise ValueError(f"Saved {name} split contains an invalid ID")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Saved {name} split contains duplicate IDs")
    if set(split["train"]).intersection(split["validation"]):
        raise ValueError("Saved train and validation splits overlap")


def validate_split(df: pd.DataFrame, split) -> None:
    """Reject duplicate, overlapping, missing, or extraneous checkpoint split IDs."""
    _validate_ids(df)
    _validate_split_lists(split)
    expected = set(df["id"])
    actual = set(split["train"]).union(split["validation"])
    if actual != expected:
        raise ValueError(
            "Saved split does not exactly cover the training CSV: "
            f"{len(expected - actual)} missing IDs, {len(actual - expected)} unexpected IDs"
        )


class TextDataset(Dataset):
    """Keep text untokenized so each loader batch can be dynamically padded."""

    def __init__(self, texts, labels=None):
        self.texts = ["" if pd.isna(text) else str(text) for text in texts]
        self.labels = (
            None
            if labels is None
            else torch.as_tensor(np.asarray(labels), dtype=torch.float32)
        )
        if self.labels is not None and self.labels.shape != (
            len(self.texts),
            len(LABELS),
        ):
            raise ValueError(
                f"Dataset labels must have shape ({len(self.texts)}, {len(LABELS)})"
            )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        item = {"text": self.texts[index]}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


def collator(tokenizer, max_length: int):
    """Return a tokenizer-backed dynamic-padding collator, retaining targets."""
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length <= 0
    ):
        raise ValueError("Maximum sequence length must be a positive integer")

    def collate(batch):
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        encoded = tokenizer(
            [item["text"] for item in batch],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        has_labels = ["labels" in item for item in batch]
        if any(has_labels) and not all(has_labels):
            raise ValueError("A batch cannot mix labeled and unlabeled examples")
        if all(has_labels):
            encoded["labels"] = torch.stack(
                [torch.as_tensor(item["labels"], dtype=torch.float32) for item in batch]
            )
        return encoded

    return collate


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid device {value!r}; use auto, cpu, cuda, or cuda:INDEX"
        ) from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only single-device CPU and CUDA execution are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested but is unavailable in this Python environment"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {device.index} is unavailable; found {torch.cuda.device_count()} devices"
            )
    elif device.index is not None:
        raise ValueError("CPU device indices are unsupported; use 'cpu'")
    return device


def seed_everything(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng(generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
        "generator": generator.get_state().clone(),
    }


def restore_rng(state: dict, generator: torch.Generator) -> None:
    required = {"python", "numpy", "cpu", "cuda", "generator"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError(
            "Checkpoint has incomplete Python/NumPy/Torch/shuffle RNG state"
        )
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["cpu"].cpu())
        generator.set_state(state["generator"].cpu())
        if torch.cuda.is_available() and state["cuda"]:
            saved_count = len(state["cuda"])
            device_count = torch.cuda.device_count()
            if saved_count != device_count:
                warnings.warn(
                    f"Checkpoint has {saved_count} CUDA RNG states but this environment has "
                    f"{device_count} devices; cross-device resume is not bitwise reproducible.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            for index, cuda_state in enumerate(state["cuda"][:device_count]):
                torch.cuda.set_rng_state(cuda_state.cpu(), device=index)
    except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
        raise ValueError(f"Invalid checkpoint RNG state: {exc}") from exc


def package_versions() -> dict:
    versions = {"python": platform.python_version(), "cuda": torch.version.cuda}
    for name in [
        "torch",
        "transformers",
        "pandas",
        "numpy",
        "scikit-learn",
        "iterative-stratification",
        "tqdm",
    ]:
        versions[name] = importlib.metadata.version(name)
    return versions


def _validate_snapshot(snapshot, name: str) -> None:
    if not isinstance(snapshot, dict):
        raise ValueError(f"Checkpoint {name} snapshot must be a dictionary")
    missing = _SNAPSHOT_FIELDS.difference(snapshot)
    if missing:
        raise ValueError(
            f"Checkpoint {name} snapshot is missing fields: {', '.join(sorted(missing))}"
        )
    if not isinstance(snapshot["labels"], list) or snapshot["labels"] != LABELS:
        raise ValueError(f"Checkpoint {name} label order must be exactly {LABELS!r}")
    for field in [
        "config",
        "model_config",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "rng",
        "versions",
    ]:
        if not isinstance(snapshot[field], dict):
            raise ValueError(f"Checkpoint {name}.{field} must be a dictionary")
    max_length = snapshot["config"].get("max_length")
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length <= 0
    ):
        raise ValueError(f"Checkpoint {name} must specify a positive config.max_length")
    if not isinstance(snapshot["model_config"].get("model_type"), str):
        raise ValueError(f"Checkpoint {name} model_config is missing model_type")
    for field, minimum in [
        ("epoch", 1),
        ("global_step", 0),
        ("best_epoch", 1),
        ("bad_epochs", 0),
    ]:
        value = snapshot[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(
                f"Checkpoint {name}.{field} must be an integer >= {minimum}"
            )
    if snapshot["best_epoch"] > snapshot["epoch"]:
        raise ValueError(
            f"Checkpoint {name} best epoch cannot follow its completed epoch"
        )
    try:
        valid_loss = math.isfinite(snapshot["best_loss"]) and snapshot["best_loss"] >= 0
    except TypeError:
        valid_loss = False
    if not valid_loss:
        raise ValueError(f"Checkpoint {name} best_loss must be finite and nonnegative")
    digest = snapshot["fingerprint"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError(
            f"Checkpoint {name} training fingerprint must be a SHA256 hex digest"
        )
    _validate_split_lists(snapshot["split"])
    history = snapshot["history"]
    if not isinstance(history, list) or len(history) != snapshot["epoch"]:
        raise ValueError(f"Checkpoint {name} history must cover every completed epoch")
    for epoch, row in enumerate(history, start=1):
        if (
            not isinstance(row, dict)
            or set(HISTORY_FIELDS).difference(row)
            or row["epoch"] != epoch
        ):
            raise ValueError(
                f"Checkpoint {name} contains an invalid history row for epoch {epoch}"
            )
        for field in ["train_loss", "validation_loss", "lr_used", "next_lr"]:
            try:
                valid = math.isfinite(row[field]) and row[field] >= 0
            except TypeError:
                valid = False
            if not valid:
                raise ValueError(
                    f"Checkpoint {name} history {field} must be finite and nonnegative"
                )
    if not {"python", "numpy", "cpu", "cuda", "generator"}.issubset(snapshot["rng"]):
        raise ValueError(f"Checkpoint {name} has incomplete RNG state")


def _validate_payload(payload) -> None:
    if (
        not isinstance(payload, dict)
        or type(payload.get("version")) is not int
        or payload["version"] != 1
    ):
        raise ValueError("Unsupported checkpoint format: expected version 1")
    if "current" not in payload or "best" not in payload:
        raise ValueError(
            "Checkpoint must contain both current and historical best snapshots"
        )
    _validate_snapshot(payload["current"], "current")
    _validate_snapshot(payload["best"], "best")
    current, best = payload["current"], payload["best"]
    if best["epoch"] != current["best_epoch"] or best["best_epoch"] != best["epoch"]:
        raise ValueError(
            "Checkpoint historical best snapshot does not match the saved best epoch"
        )
    for field in ["best_loss", "fingerprint", "split", "config"]:
        if current[field] != best[field]:
            raise ValueError(f"Checkpoint current/best snapshots disagree on {field}")


def load_checkpoint(path) -> tuple[dict, Path]:
    """Load a versioned, trusted local checkpoint directory (including best/)."""
    if "://" in str(path):
        raise ValueError(
            "Checkpoint must be a trusted local directory, not a URL or Hub identifier"
        )
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Trusted local checkpoint directory does not exist: {directory}"
        )
    directory = directory.resolve()
    checkpoint = directory / _CHECKPOINT_FILE
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint directory is missing {_CHECKPOINT_FILE}: {directory}"
        )
    warnings.warn(
        f"Loading trusted local checkpoint {checkpoint}. PyTorch pickle data can execute code; "
        "never load an untrusted checkpoint.",
        RuntimeWarning,
        stacklevel=2,
    )
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(
            f"Cannot load trusted local checkpoint {checkpoint}: {exc}"
        ) from exc
    _validate_payload(payload)
    return payload, directory


def build_model(snapshot: dict, device: torch.device):
    """Reconstruct architecture and weights entirely from the saved configuration."""
    config_values = dict(snapshot["model_config"])
    model_type = config_values.pop("model_type", None)
    if not isinstance(model_type, str):
        raise ValueError("Checkpoint model_config must contain model_type")
    try:
        config = AutoConfig.for_model(model_type, **config_values)
        if config.num_labels != len(LABELS):
            raise ValueError(
                f"Model configuration must have exactly {len(LABELS)} outputs"
            )
        model = AutoModelForSequenceClassification.from_config(
            config, trust_remote_code=False
        )
        model.load_state_dict(snapshot["model_state"], strict=True)
        return model.to(device)
    except torch.OutOfMemoryError as exc:
        raise ValueError(
            "Checkpoint model does not fit in device memory; try --device cpu"
        ) from exc
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise ValueError(f"Cannot reconstruct checkpoint model: {exc}") from exc


def load_inference(path, device_string: str):
    device = choose_device(device_string)
    payload, directory = load_checkpoint(path)
    snapshot = payload["current"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(
            f"Cannot load local checkpoint tokenizer from {directory}: {exc}"
        ) from exc
    model = build_model(snapshot, device)
    model.eval()
    return model, tokenizer, snapshot, device, directory


def save_checkpoint(
    output_dir, name: str, tokenizer, current: dict, best: dict
) -> Path:
    """Atomically commit an immutable directory; never replace an existing epoch."""
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name in {".", ".."}
    ):
        raise ValueError("Checkpoint name must be a single nonempty directory name")
    payload = {"version": 1, "current": current, "best": best}
    _validate_payload(payload)
    parent = Path(output_dir).expanduser() / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / name
    if os.path.lexists(destination):
        raise FileExistsError(
            f"Refusing to overwrite immutable checkpoint: {destination}"
        )
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=parent))
    try:
        tokenizer.save_pretrained(temporary)
        with (temporary / _CHECKPOINT_FILE).open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(destination):
            raise FileExistsError(
                f"Refusing to overwrite immutable checkpoint: {destination}"
            )
        os.rename(temporary, destination)
    except (OSError, RuntimeError) as exc:
        raise OSError(
            f"Unable to commit checkpoint {destination}; prior checkpoints are unchanged: {exc}"
        ) from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def point_best(output_dir, checkpoint_path) -> None:
    """Atomically replace best with a relative symlink to a committed checkpoint."""
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if (
        not checkpoint_path.is_dir()
        or not (checkpoint_path / _CHECKPOINT_FILE).is_file()
    ):
        raise ValueError(
            f"Best pointer target is not a committed checkpoint: {checkpoint_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "best"
    if os.path.lexists(destination) and not destination.is_symlink():
        raise FileExistsError(
            f"Refusing to replace a non-symlink best location: {destination}"
        )
    temporary = output_dir / f".best-{uuid.uuid4().hex}.tmp"
    try:
        temporary.symlink_to(
            os.path.relpath(checkpoint_path, output_dir), target_is_directory=True
        )
        os.replace(temporary, destination)
    except OSError as exc:
        raise OSError(
            f"Unable to update best checkpoint pointer {destination}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_history(output_dir, history) -> None:
    """Atomically synchronize losses.csv; committed checkpoint history is authoritative."""
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=".losses-",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(history)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_dir / "losses.csv")
    except (OSError, ValueError) as exc:
        raise OSError(
            f"Unable to write loss history {output_dir / 'losses.csv'}: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def predict_texts(
    model,
    tokenizer,
    texts,
    max_length: int,
    batch_size: int,
    device: torch.device,
    description: str = "Predicting",
) -> np.ndarray:
    """Predict six independent sigmoid probabilities in the original text order."""
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("Prediction batch size must be a positive integer")
    dataset = TextDataset(texts)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator(tokenizer, max_length),
    )
    if not len(dataset):
        return np.empty((0, len(LABELS)), dtype=np.float32)
    probabilities = []
    model.eval()
    try:
        with torch.inference_mode():
            for batch in tqdm(loader, desc=description, file=sys.stderr):
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = model(**batch).logits
                if logits.ndim != 2 or logits.shape[1] != len(LABELS):
                    raise ValueError(
                        f"Model must return logits with shape (batch, {len(LABELS)})"
                    )
                values = logits.sigmoid().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError(
                        "Model returned non-finite prediction probabilities"
                    )
                probabilities.append(values)
    except torch.OutOfMemoryError as exc:
        raise ValueError(
            "Prediction ran out of memory; reduce test --batch-size, use --device cpu, "
            "or create a checkpoint trained with a smaller --max-length"
        ) from exc
    return np.concatenate(probabilities, axis=0)
