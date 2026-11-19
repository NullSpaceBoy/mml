#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/workspace}
VENV_DIR=${VENV_DIR:-$PROJECT_DIR/.venv}
: ${TRAIN_CMD:?Set TRAIN_CMD to your training command}
: ${TEST_CMD:?Set TEST_CMD to your evaluation/testing command}
cd $PROJECT_DIR
if [[ ! -x $VENV_DIR/bin/python ]]; then
    echo "No usable virtual environment at $VENV_DIR. Run setup.sh first." >&2
    exit 1
fi

printf '\n== Training ==\n'
bash -euo pipefail -c "$VENV_DIR/bin/python $TRAIN_CMD"
printf '\n== Evaluation ==\n'
bash -euo pipefail -c "$VENV_DIR/bin/python $TEST_CMD"
printf '\nTraining and evaluation completed successfully.\n'

if [[ ${STOP_POD:-1} == 1 ]]; then
    : ${RUNPOD_POD_ID:?"RUNPOD_POD_ID required to stop the pod"}
    : ${RUNPOD_API_KEY:?"RUNPOD_API_KEY required to stop the pod"}
    echo 'Requesting RunPod pod stop...'
    # RunPod GraphQL podStop mutation; stopping retains the pod and its network volume.
    runpodctl pod stop $RUNPOD_POD_ID
    printf 'Pod stop requested.\n'
fi
