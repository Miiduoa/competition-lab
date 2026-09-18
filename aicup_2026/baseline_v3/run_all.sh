#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== AI CUP 2026 baseline_v3 (LOCAL holdout, not leaderboard) ==="
python3 task1_v3.py "$@"
python3 task2_v3.py "$@"
echo "=== done ==="
