#!/bin/bash

# Default to "dl-env" if no path is provided
VENV_PATH=${1:-"dl-processing-pipeline/training/dl-env"}

# Check if the specified virtual environment directory exists
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
    echo "Virtual environment created at $VENV_PATH."
fi

source "$VENV_PATH/bin/activate"
pip install -r dl-processing-pipeline/training/requirements.txt
echo "Dependencies installed in $VENV_PATH."