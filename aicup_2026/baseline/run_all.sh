#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== AI CUP 2026 Go baselines ==="
echo "cwd: $(pwd)"
python3 task1_baseline.py "$@"
python3 task2_baseline.py "$@"
echo "=== done; see ../outputs/ ==="
ls -la ../outputs/task*_baseline_*
