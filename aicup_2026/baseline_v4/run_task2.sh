#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHONUNBUFFERED=1 python3 task2_v4.py --reuse-cache "${1:-25000}" --max-games-per-rank "${1:-25000}" --max-examples-per-player 6
