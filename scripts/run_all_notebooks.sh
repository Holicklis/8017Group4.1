#!/usr/bin/env bash
# Execute preprocessing → chatbot notebooks in order (from repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/notebooks"
export PYTHONUNBUFFERED=1
LOG="$ROOT/pipeline.log"
: >"$LOG"
for nb in 01_preprocessing 02_classification 03_regression 04_unsupervised 05_chatbot; do
  echo "==== $(date -Iseconds) START $nb ====" | tee -a "$LOG"
  python3 -m jupyter nbconvert \
    --to notebook \
    --execute "${nb}.ipynb" \
    --output "${nb}_executed.ipynb" \
    --ExecutePreprocessor.timeout=-1 \
    2>&1 | tee -a "$LOG"
  echo "==== $(date -Iseconds) END $nb ====" | tee -a "$LOG"
done
echo "All notebooks finished OK." | tee -a "$LOG"
