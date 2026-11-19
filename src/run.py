"""Print six sigmoid probabilities as JSON for one comment, including empty text.

Example: python experiments/bert-baseline/src/run.py --checkpoint runs/bert/best --text 'hello'
Model/tokenizer assets and preprocessing settings are loaded entirely locally.
Checkpoints contain pickle data: load only artifacts from a trusted source.
Diagnostic output goes to stderr; stdout contains only the JSON object.
"""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from common import load_inference, predict_texts


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
        "--text", required=True, help="Comment to score; an empty string is allowed"
    )
    parser.add_argument(
        "--device", default="auto", help="Torch device, e.g. auto, cpu, cuda, cuda:0"
    )
    args = parser.parse_args(argv)
    try:
        # Libraries may log during model construction as well as prediction.
        with redirect_stdout(sys.stderr):
            model, tokenizer, snapshot, device, _ = load_inference(
                args.checkpoint, args.device
            )
            probabilities = predict_texts(
                model,
                tokenizer,
                [args.text],
                snapshot["config"]["max_length"],
                1,
                device,
                description="Predicting",
            )
        output = dict(
            zip(snapshot["labels"], (float(value) for value in probabilities[0]))
        )
        print(json.dumps(output, allow_nan=False))
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
