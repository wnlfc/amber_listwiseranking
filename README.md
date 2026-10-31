# AMBER

Code for the current AMBER main method: HOM candidate views, view-conditioned adaptive query scheduling, and Elo aggregation. This release retains the filenames and implementation used for the experiments.

## Files

```text
elo_rerank.py                     HOM view, Elo ranking, adaptive scheduler, and entry point
view_aware_listwise_reranker.py   Image preparation and VLM requests
prompts.py                        Listwise prompts and response parser
config/protocol.json              Main experiment settings and input-path conventions
examples/                         CIRCO, CIRR, and PhotoBench launchers
make_submission.py                CIRCO/CIRR official JSON conversion
evaluate_photobench.py            Local PhotoBench metrics
test_evaluation.py                Evaluation-format checks
```

The single algorithm entry point combines the original experiment core with the current `pair_marginal_view` scheduler. The examples use `--mode scheduler-only` with HOM views. The dataset-specific prompt and image code remain in their original files.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Serve Qwen3-VL-8B-Instruct through an OpenAI-compatible API, then set any required credentials in the environment:

```bash
export AMBER_API_URL=http://localhost:8005/v1
export AMBER_MODEL=qwen3-vl-8b-instruct
# For an authenticated server:
export AMBER_API_KEY=your_api_key
```

## Run

Provide the first-stage retrieval JSON and an image directory. For CIRCO:

```bash
RETRIEVAL_JSON=/path/to/circo_retrieval.json \
IMAGE_DIR=/path/to/circo_images \
AMBER_BUDGET=24 \
bash examples/run_circo.sh
```

For CIRR, use `examples/run_cirr.sh`. Both scripts run in the foreground and then write an official submission JSON beside the JSONL trace. Set `OUTPUT` to a new path under this repository when running another configuration. `AMBER_SEED` defaults to 42; `AMBER_MAX_WORKERS` defaults to 32.

For a PhotoBench split:

```bash
AMBER_DATASET=photobench-subset03-cn \
RETRIEVAL_JSON=/path/to/subset03-cn.json \
IMAGE_DIR=/path/to/photo03 \
AMBER_BUDGET=24 \
bash examples/run_photobench.sh
```

Run the six PhotoBench subset/language combinations separately, then aggregate each language using `evaluate_photobench.py` with three paired input and output files. The evaluator reports Recall and NDCG at 1, 5, 10, 20, and 50. Empty ground-truth queries remain in the denominator. PhotoBench defaults to six concurrent requests.

The underlying entry point also accepts direct commands:

```bash
python elo_rerank.py --mode scheduler-only --dataset circo \
  --retrieval /path/to/circo_retrieval.json \
  --image_dir /path/to/circo_images \
  --budget 24 --seed 42 \
  --output outputs/circo/amber_circo_b24_s42.jsonl
```

`--print_command` checks and displays the resolved configuration without calling the model. Use `python elo_rerank.py --help` for runtime options.

## Main settings

The [protocol file](config/protocol.json) records the experiment defaults: 50 retrieved candidates, views of 10, initial one call per query, adaptive batches of 16, and a per-query cap of `min(50, 2 × budget)`. Elo uses base 1500, initialization range 400, scale 400, and update size 16. HOM uses margin scale 50 and rank-neighborhood radius 2. The default budget is 24 calls per query; use 8 for the low-budget experiment. Presentation seed is 42, temperature 0, and the output cap is 512 tokens.

The image rules follow the experiment scripts: CIRCO uses original image bytes, CIRR resizes to 768 pixels on the long side, and PhotoBench to 1120 pixels. Invalid output permutations follow the original strict fallback: the call consumes budget and updates pair exposure, while Elo and candidate observation counts stay unchanged. `enable_thinking=false` is included in the API request; some servers ignore that field.

The input JSON contains `cases`, each with `query_id`, `query`, `retrieved_top50` (or `retrieved_top100`), and a `reference` image for CIRCO/CIRR. PhotoBench cases have text queries and ground truth. Image filenames are resolved against `--image_dir`. The retrieval JSON and images are provided separately.

## Evaluation

The launchers call the conversion or evaluation tool automatically. For an existing CIRCO run:

```bash
python make_submission.py --dataset circo \
  --retrieval /path/to/circo_retrieval.json \
  --rerank outputs/circo/amber_circo_b24_s42.jsonl \
  --output submissions/circo_b24_s42.json
```

Use `--dataset cirr` for CIRR. Test the local evaluation format with:

```bash
python -m unittest test_evaluation.py
```

`data/`, `images/`, `outputs/`, and local API configuration are ignored by Git. The repository contains code and documentation only.
