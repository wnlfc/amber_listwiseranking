#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${RETRIEVAL_JSON:?Set RETRIEVAL_JSON to the CIRR retrieval JSON}"
output="${OUTPUT:-outputs/cirr/amber_cirr_b${AMBER_BUDGET:-24}_s${AMBER_SEED:-42}.jsonl}"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" elo_rerank.py --mode scheduler-only --dataset cirr \
  --retrieval "$RETRIEVAL_JSON" --image_dir "${IMAGE_DIR:-images/cirr}" \
  --budget "${AMBER_BUDGET:-24}" --seed "${AMBER_SEED:-42}" \
  --max_workers "${AMBER_MAX_WORKERS:-32}" --output "$output"
"$python_bin" make_submission.py --dataset cirr --retrieval "$RETRIEVAL_JSON" \
  --rerank "$output" --output "${output%.jsonl}.json"
