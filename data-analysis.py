#!/usr/bin/env python3
"""Count comments exceeding selected BERT token lengths in train and test CSVs."""

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def count_long_comments(path, tokenizer):
    frame = pd.read_csv(path, usecols=["comment_text"], keep_default_na=False)
    token_counts = [
        len(tokenizer.encode(str(comment), add_special_tokens=True))
        for comment in frame["comment_text"]
    ]
    return (
        len(token_counts),
        sum(count > 128 for count in token_counts),
        sum(count > 256 for count in token_counts),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    for name, path in (("Train", args.train_csv), ("Test", args.test_csv)):
        total, over_128, over_256 = count_long_comments(path, tokenizer)
        print(f"{name} ({total} comments):")
        print(f"  Exceeding 128 tokens: {over_128}")
        print(f"  Exceeding 256 tokens: {over_256}")


if __name__ == "__main__":
    main()
