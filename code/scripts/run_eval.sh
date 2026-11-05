#!/usr/bin/env bash
# Default: camera-ready pipeline. Remaining arguments go to run_pipeline.py.
set -euo pipefail
exec python3 "$(dirname "$0")/run_pipeline.py" "$@"
