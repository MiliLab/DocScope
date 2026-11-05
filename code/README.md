# DocScope evaluation

## Install and configure

Use Python 3.10+ and install dependencies:

```bash
pip install -r requirements.txt
```

Install Poppler (`pdftoppm`) for page rendering. Edit `models.yaml` to configure the inference model's `id`, `backend`, `model`, `api_key_env` and `base_url`, then export its API key.

The pipelines download a pinned Hugging Face dataset snapshot, including PDFs, on first use. Use `--dataset-dir /path/to/snapshot` for a local copy. The benchmark SHA-256 is checked before any inference. Run from a full Git clone (the legacy entry reads the frozen code commit).

## Camera-ready (default)

```bash
export JUDGE_API_KEY=...
export JUDGE_BASE_URL=...
export JUDGE_MODEL=qwen3.6-plus
export BBOX_JUDGE_API_KEY=...
export BBOX_JUDGE_BASE_URL=...
export BBOX_JUDGE_MODEL=gpt-5.5
export ATTRIBUTION_JUDGE_API_KEY=...
export ATTRIBUTION_JUDGE_BASE_URL=...
export ATTRIBUTION_JUDGE_MODEL=qwen3.8-max
bash scripts/run_eval.sh <model_id>
```

Region and attribution credentials/endpoints fall back to `JUDGE_API_KEY` and `JUDGE_BASE_URL`. Model names above are the defaults; adjust provider-specific aliases as needed. Different judge models or fresh generations may produce different scores from the paper.

Flow: pinned new data → inference → answer verification, individual predicted-box matching, multi-evidence fact coverage and citation attribution → nine camera-ready metrics. No private historical judgment artifacts are required.

Output: `results/camera-ready/<model_id>/test/summary.json`. Scores are percentages; a metric with an empty denominator is `null`. Per-question judge responses and usage are stored in `judgments/`. Missing or invalid judgments fail the run without producing a final summary. Completed question judgments are cached; rerunning retries unfinished questions.

## Legacy

```bash
# Set these to the judge configuration used for your legacy experiment.
export JUDGE_API_KEY=...
export JUDGE_BASE_URL=...
export JUDGE_MODEL=...
export BBOX_JUDGE_MODEL=...
bash scripts/run_eval_legacy.sh <model_id>
```

Flow: pinned old data → inference → original page, region, fact and answer judges → original metric summaries. The runtime uses code commit `ac0de26b050d4f51de0407c45b5cf3fdf245e1bd` and dataset revision `4326db57d779829a44e46849fed0ebbe782f2e16`. The original scoring and aggregation implementations are unchanged. Output: `results/legacy/<model_id>/test/summary.json` (legacy units and schema).

## Shared options

```bash
bash scripts/run_eval.sh <model_id> --limit 2
bash scripts/run_eval_legacy.sh <model_id> --limit 2
bash scripts/run_eval.sh <model_id> --split dev --output-root /path/to/runs
bash scripts/run_eval.sh <model_id> --prepare-only
```

Both pipelines default to the test split. `--limit N` selects the first N questions before inference and is intended for smoke tests. Version, split and limit have separate output directories. `run.json` records data revision/hash, code version, model and judge configuration. Configuration changes require a different output root. API keys are not written to the manifest.

For the camera-ready definitions and archived main-table calculator, see [camera-ready evaluation](eval/camera_ready/README.md).

## Offline tests

```bash
python3 -m unittest discover -s tests -v
```
