#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== Task1 v2 ==="
python3 task1_v2.py "$@"
echo "=== Task2 v2 ==="
python3 task2_v2.py "$@"
echo "=== done ==="
