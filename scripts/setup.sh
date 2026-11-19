#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(pwd -P)
VENV_DIR=${VENV_DIR:-$PROJECT_DIR/.venv}
mkdir -p $VENV_DIR
VENV_DIR=$(cd $VENV_DIR && pwd -P)
cd $PROJECT_DIR

if ! command -v python3.12 >/dev/null 2>&1; then
    echo 'Python 3.12 is required.' >&2
    exit 1
fi
if ! dpkg-query -W -f='${Status}' python3.12-venv 2>/dev/null | grep -qx "install ok installed"; then
    apt-get update
    apt-get install -y python3.12-venv
fi
if [[ ! -f $VENV_DIR/pyvenv.cfg ]]; then
    python -m venv --system-site-packages $VENV_DIR
fi
$VENV_DIR/bin/pip install -r requirements.txt
$VENV_DIR/bin/pip check
printf 'Virtual environment ready: %s\n' $VENV_DIR
