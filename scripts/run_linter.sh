#!/bin/bash
echo "Running Ruff linter..."
source ./scripts/setup_env.sh
ruff check dl-processing-pipeline/ --config pyproject.toml --fix
