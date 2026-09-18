#!/usr/bin/env bash
# 打包 submission.zip（main.py 必須在 ZIP 根目錄）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/submission_src"
OUT_DIR="$ROOT/submission"
OUT_ZIP="$OUT_DIR/submission.zip"
mkdir -p "$OUT_DIR"
rm -f "$OUT_ZIP"
python3 - << PY
from pathlib import Path
import zipfile
src = Path("$SRC")
out = Path("$OUT_ZIP")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(src.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            arc = p.relative_to(src).as_posix()
            zf.write(p, arcname=arc)
            print(" +", arc)
print("Wrote", out, "size=", out.stat().st_size)
PY
python3 - << PY
import zipfile
z = zipfile.ZipFile("$OUT_ZIP")
print("--- archive listing ---")
for i in z.infolist():
    print(f"{i.file_size:8d}  {i.filename}")
names = {i.filename for i in z.infolist()}
assert "main.py" in names, "main.py must be at ZIP root"
print("OK: main.py at root")
PY
