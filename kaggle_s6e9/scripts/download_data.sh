#!/usr/bin/env bash
# 需已設定 ~/.kaggle/kaggle.json 且已同意競賽規則
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data"
kaggle competitions download -c playground-series-s6e9 -p "$ROOT/data"
unzip -o "$ROOT/data"/*.zip -d "$ROOT/data"
echo "Done. Files in $ROOT/data:"
ls -la "$ROOT/data"
