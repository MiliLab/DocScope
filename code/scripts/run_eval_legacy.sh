#!/usr/bin/env bash
# Frozen legacy code and dataset; uses the original judge and metric implementations.
set -euo pipefail
exec python3 "$(dirname "$0")/run_pipeline.py" "$@" --version legacy
