#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
dataset="${AMBER_DATASET:?Set AMBER_DATASET, e.g. photobench-subset03-cn}"
case "$dataset" in
  photobench-subset03-*) image="${IMAGE_DIR:-${PHOTOBENCH_IMAGE_ROOT:-images/photobench}/photo03}" ;;
  photobench-subset11-*) image="${IMAGE_DIR:-${PHOTOBENCH_IMAGE_ROOT:-images/photobench}/photo11}" ;;
  photobench-subset12-*) image="${IMAGE_DIR:-${PHOTOBENCH_IMAGE_ROOT:-images/photobench}/photo12}" ;;
  *) echo "Unsupported PhotoBench split: $dataset" >&2; exit 2 ;;
esac
: "${RETRIEVAL_JSON:?Set RETRIEVAL_JSON to this PhotoBench split}"
output="${OUTPUT:-outputs/photobench/${dataset}_b${AMBER_BUDGET:-24}_s${AMBER_SEED:-42}.jsonl}"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" elo_rerank.py --mode scheduler-only --dataset "$dataset" \
  --retrieval "$RETRIEVAL_JSON" --image_dir "$image" \
  --budget "${AMBER_BUDGET:-24}" --seed "${AMBER_SEED:-42}" \
  --max_workers "${AMBER_MAX_WORKERS:-6}" --output "$output"
"$python_bin" evaluate_photobench.py --retrieval "$RETRIEVAL_JSON" --rerank "$output"
