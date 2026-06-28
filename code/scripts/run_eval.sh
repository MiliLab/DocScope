#!/usr/bin/env bash
# Run the full eval pipeline for one model from models.yaml.
#
# Usage:
#   scripts/run_eval.sh <model_id> [LIMIT]
#
# Required env vars:
#   JUDGE_API_KEY   — API key for the judge endpoint
#   JUDGE_BASE_URL  — base URL of the judge (OpenAI-compatible)
#   JUDGE_MODEL     — judge model name (default: gpt-4o-mini)
#
# Steps:
#   1. infer/run_infer.py   → results/<id>/infer.jsonl
#   2. eval/score_pages.py  → results/<id>/pages.jsonl
#   3. eval/score_bbox.py   → results/<id>/bbox.jsonl
#   4. eval/score_facts.py  → results/<id>/facts.jsonl
#   5. eval/score_answer.py → results/<id>/answer.jsonl
#   6. scripts/summarize_eval.py → results/<id>/summary.json
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model_id> [LIMIT]" >&2; exit 2
fi
MODEL_ID="$1"
LIMIT="${2:-${LIMIT:-0}}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

read -r BACKEND MODEL API_KEY_ENV BASE_URL BENCHMARK OUTPUT_ROOT INFER_CC EVAL_CC < <(
  python3 - "$MODEL_ID" <<'PY'
import sys, yaml
mid = sys.argv[1]
cfg = yaml.safe_load(open("models.yaml"))
d   = cfg.get("defaults") or {}
m   = next((x for x in cfg["models"] if x["id"] == mid), None)
if m is None:
    sys.stderr.write(f"unknown model id: {mid}\n"); sys.exit(2)
print(
    m["backend"], m["model"], m["api_key_env"], m["base_url"],
    d.get("benchmark", "benchmark.json"),
    d.get("output_root", "results"),
    d.get("infer_concurrency", 4),
    d.get("eval_concurrency", 32),
)
PY
)

EXTRA_ARGS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && EXTRA_ARGS+=("$line")
done < <(python3 - "$MODEL_ID" <<'PY'
import sys, yaml
mid = sys.argv[1]
cfg = yaml.safe_load(open("models.yaml"))
m = next(x for x in cfg["models"] if x["id"] == mid)
for a in (m.get("extra_args") or []):
    print(a)
PY
)

OUT_DIR="$OUTPUT_ROOT/$MODEL_ID"
mkdir -p "$OUT_DIR" "$OUT_DIR/imgs"

LIMIT_ARG=()
[[ "$LIMIT" != "0" ]] && LIMIT_ARG=(--limit "$LIMIT")

echo "════════════════════════════════════════════════════════════"
echo "[$(date +%H:%M:%S)] $MODEL_ID  backend=$BACKEND  model=$MODEL"
echo "  output=$OUT_DIR"
echo "════════════════════════════════════════════════════════════"

echo "── 1/6: inference ──"
python3 infer/run_infer.py \
  --backend "$BACKEND" --model "$MODEL" \
  --api_key_env "$API_KEY_ENV" --base_url "$BASE_URL" \
  --benchmark "$BENCHMARK" \
  --concurrency "$INFER_CC" \
  --output "$OUT_DIR/infer.jsonl" \
  ${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}} \
  ${LIMIT_ARG[@]+${LIMIT_ARG[@]}}

echo "── 2/6: page recall ──"
python3 eval/score_pages.py \
  --infer "$OUT_DIR/infer.jsonl" \
  --output "$OUT_DIR/pages.jsonl" \
  --benchmark "$BENCHMARK"

echo "── 3/6: bbox grounding ──"
python3 eval/score_bbox.py \
  --infer "$OUT_DIR/infer.jsonl" \
  --output "$OUT_DIR/bbox.jsonl" \
  --benchmark "$BENCHMARK" \
  --imgs-dir "$OUT_DIR/imgs" \
  --concurrency "$EVAL_CC" \
  ${LIMIT_ARG[@]+${LIMIT_ARG[@]}}

echo "── 4/6: fact consistency ──"
python3 eval/score_facts.py \
  --infer "$OUT_DIR/infer.jsonl" \
  --output "$OUT_DIR/facts.jsonl" \
  --benchmark "$BENCHMARK" \
  --concurrency "$EVAL_CC" \
  ${LIMIT_ARG[@]+${LIMIT_ARG[@]}}

echo "── 5/6: answer verification ──"
python3 eval/score_answer.py \
  --infer "$OUT_DIR/infer.jsonl" \
  --output "$OUT_DIR/answer.jsonl" \
  --benchmark "$BENCHMARK" \
  --concurrency "$EVAL_CC" \
  ${LIMIT_ARG[@]+${LIMIT_ARG[@]}}

echo "── 6/6: summarize ──"
python3 scripts/summarize_eval.py --eval-dir "$OUT_DIR" | tee "$OUT_DIR/summary.json"
echo "✔ done: $OUT_DIR"
