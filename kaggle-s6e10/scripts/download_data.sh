#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/../data" && pwd)"
mkdir -p "$DIR"
kaggle competitions download -c playground-series-s6e10 -p "$DIR"
unzip -o "$DIR/playground-series-s6e10.zip" -d "$DIR"
echo "files in $DIR:"
ls -lh "$DIR"/*.csv
