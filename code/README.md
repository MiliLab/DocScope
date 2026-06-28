# DocScope Benchmark

A document question answering benchmark requiring fine-grained visual evidence localization in real-world PDFs.

Each question is paired with gold evidence regions (bounding boxes) and atomic facts, enabling multi-dimensional evaluation of both answer correctness and evidence grounding quality.

## Repository Structure

```
DocScope/
├── benchmark.json           # 1124 questions (split: test=730, dev=394)
├── pdfs/                    # 273 PDF documents (downloaded separately)
├── models.yaml              # model registry
├── infer/
│   ├── run_infer.py         # inference entry point
│   ├── reasoning_common.py  # PDF rendering, JSONL helpers
│   ├── reasoning_prompts.py # system prompt + citation parsing
│   ├── benchmark_io.py      # benchmark loader
│   └── backends/
│       ├── claude.py        # Anthropic Claude
│       ├── gemini.py        # Google Gemini
│       └── openai_compat.py # OpenAI and compatible APIs
├── eval/
│   ├── score_pages.py       # Step 1: page-level recall (no LLM)
│   ├── score_bbox.py        # Step 2: bounding-box grounding (multimodal judge)
│   ├── score_facts.py       # Step 3: atomic fact consistency (text judge)
│   └── score_answer.py      # Step 4: answer verification (text judge)
├── prompts/                 # judge prompt templates
├── scripts/
│   ├── run_eval.sh          # one-command full pipeline
│   └── summarize_eval.py    # aggregate results → summary.json
└── results/                 # outputs (created on first run, not committed)
```

## Installation

```bash
pip install PyMuPDF openai requests pyyaml tqdm pillow
```

For bbox evaluation, `pdftoppm` (part of `poppler-utils`) is also required:

```bash
# Ubuntu / Debian
apt-get install poppler-utils

# macOS
brew install poppler
```

## Step 1 — Download Data

The benchmark data is hosted on HuggingFace. Download `benchmark.json` and the PDF documents:

```bash
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="MiliLab/DocScope",
    repo_type="dataset",
    local_dir=".",
    allow_patterns=["benchmark.json", "pdfs/*"],
)
EOF
```

Or use the CLI:

```bash
huggingface-cli download MiliLab/DocScope \
    --repo-type dataset \
    --include "benchmark.json" "pdfs/*" \
    --local-dir .
```

After downloading, your directory should contain:
```
benchmark.json
pdfs/
  urnuuid....pdf
  ...  (273 files)
```

## Step 2 — Inference

### Set API keys

```bash
# Claude (Anthropic)
export ANTHROPIC_API_KEY=sk-ant-...

# Gemini (Google AI Studio)
export GEMINI_API_KEY=AIza...

# OpenAI
export OPENAI_API_KEY=sk-...
```

### Run inference on the test split

```bash
# Claude
python infer/run_infer.py \
    --backend claude \
    --model claude-opus-4-7 \
    --split test \
    --output results/claude-opus-4-7/infer.jsonl

# Gemini
python infer/run_infer.py \
    --backend gemini \
    --model gemini-2.5-flash-preview-04-17 \
    --split test \
    --output results/gemini-2.5-flash/infer.jsonl

# OpenAI
python infer/run_infer.py \
    --backend openai \
    --model gpt-4o \
    --split test \
    --output results/gpt-4o/infer.jsonl
```

Inference is **resumable**: re-running with the same `--output` path skips already completed `question_id`s.

### Use a custom endpoint

Any OpenAI-compatible API is supported via `--base_url` and `--api_key_env`:

```bash
python infer/run_infer.py \
    --backend openai \
    --model your-model-name \
    --base_url https://your-provider.com/v1 \
    --api_key_env YOUR_API_KEY_ENV \
    --split test \
    --output results/your-model/infer.jsonl
```

### CLI reference

```
python infer/run_infer.py --help

  --backend       claude | openai | gemini
  --model         Model name/ID
  --output        Output JSONL path
  --benchmark     Path to benchmark.json
  --split         test | dev
  --concurrency   Parallel workers
  --max_pages     Max pages per document
  --dpi           PDF render DPI
  --base_url      Override API base URL
  --api_key_env   Env var name for API key
```

### Inference output format

Each line in the output JSONL:

```json
{
  "question_id": "urnuuid...::q1",
  "model": "claude-opus-4-7",
  "model_raw": "<full model response>",
  "thinking": "<extended reasoning trace, if available>",
  "parsed": {
    "answer": "11 - 30 mmbf",
    "citations": [
      {
        "page": 51,
        "doc_page": "49",
        "bbox": [0.412, 0.215, 0.687, 0.248],
        "sentence": "The 11-30 mmbf category has 0 sawmills in Eastern Washington."
      }
    ],
    "predicted_pages": [51, 53],
    "has_answer_tag": true
  },
  "infer_usage": {
    "prompt_tokens": 45231,
    "completion_tokens": 412,
    "total_tokens": 45643
  },
  "status": "ok"
}
```

## Step 3 — Evaluation

Evaluation runs four scoring steps, the last three of which require an OpenAI-compatible judge.

### Configure judge models

**Text judge** — used by `score_facts` and `score_answer`:

```bash
export JUDGE_API_KEY=...
export JUDGE_BASE_URL=https://api.openai.com/v1
export JUDGE_MODEL=gpt-4o-mini          # optional, this is the default
```

**BBox judge** — used by `score_bbox`, must support multimodal (vision) input:

```bash
export BBOX_JUDGE_API_KEY=...           # falls back to JUDGE_API_KEY if unset
export BBOX_JUDGE_BASE_URL=https://api.openai.com/v1
export BBOX_JUDGE_MODEL=gpt-4o          # optional, defaults to JUDGE_MODEL
```

For best comparability across submissions, we recommend using the same judge model for all runs.

### Run the full evaluation pipeline

If the model is registered in `models.yaml`:

```bash
scripts/run_eval.sh claude-opus-4-7
```

Or run each step manually:

```bash
MODEL_DIR=results/claude-opus-4-7

# Step 1: page-level recall (no LLM required)
python eval/score_pages.py \
    --infer  $MODEL_DIR/infer.jsonl \
    --output $MODEL_DIR/pages.jsonl

# Step 2: bounding-box grounding (multimodal judge)
python eval/score_bbox.py \
    --infer    $MODEL_DIR/infer.jsonl \
    --output   $MODEL_DIR/bbox.jsonl \
    --imgs-dir $MODEL_DIR/imgs

# Step 3: atomic fact consistency (text judge)
python eval/score_facts.py \
    --infer  $MODEL_DIR/infer.jsonl \
    --output $MODEL_DIR/facts.jsonl

# Step 4: answer verification (text judge)
python eval/score_answer.py \
    --infer  $MODEL_DIR/infer.jsonl \
    --output $MODEL_DIR/answer.jsonl

# Summarize all steps
python scripts/summarize_eval.py --eval-dir $MODEL_DIR
```

### Evaluation metrics

| Step | Output | Metric |
|------|--------|--------|
| `score_pages` | `pages.jsonl` | Page recall / precision / F1 (gold vs. predicted pages) |
| `score_bbox` | `bbox.jsonl` | Per-evidence coverage: `covered` / `imprecise` / `not_covered` |
| `score_facts` | `facts.jsonl` | Per-fact consistency: `consistent` / `not_consistent` |
| `score_answer` | `answer.jsonl` | Answer accuracy (semantic equivalence to gold answer) |

All steps are resumable — rerunning appends only missing entries.

### Summary output

`summarize_eval.py` produces a `summary.json`:

```json
{
  "pages":  {"macro": {"f1": 0.86}, "micro": {"f1": 0.90}},
  "bbox":   {"n": 1843, "by_label": {"covered": 1102, "imprecise": 312, "not_covered": 429}},
  "facts":  {"n": 5621, "by_label": {"consistent": 3814, "not_consistent": 1807}},
  "answer": {"n_judged": 730, "n_correct": 487, "accuracy": 0.667}
}
```

## Benchmark Format

`benchmark.json` is a flat JSON array. Each record:

```json
{
  "id": "urnuuid...::q1",
  "split": "test",
  "question": "Which firm size category at or above 10 mmbf is absent for Eastern Washington?",
  "answer": {
    "answer_text": "11 - 30 mmbf",
    "is_answerable": true
  },
  "pdf": {
    "doc_id_str": "urnuuid00588fd1-b7d1-41dc-9131-7832d01dc6b3"
  },
  "evidences": [
    {"local_id": "e1", "page": 51, "bbox": [548.4, 184.0, 653.3, 287.7], "element_type": "text"},
    {"local_id": "e2", "page": 53, "bbox": [167.1, 355.4, 803.4, 409.6], "element_type": "text"}
  ],
  "facts": [
    {"local_id": "f1", "evidence_local_id": "e1", "text_description": "In Table 7, the 11-30 mmbf category has 0 sawmills for Eastern Washington."},
    {"local_id": "f2", "evidence_local_id": "e2", "text_description": "In Table 10, the 11-30 mmbf category has 0 sawmills for the East region."}
  ],
  "extract_class": "class2"
}
```

**Field notes:**
- `pdf.doc_id_str` — matches the filename in `pdfs/` (e.g., `pdfs/urnuuid....pdf`)
- `evidences[].bbox` — pixel coordinates at **144 DPI** (x1, y1, x2, y2, origin top-left)
- `evidences[].local_id` — links to `facts[].evidence_local_id`
- `split` — `"test"` (730 questions) or `"dev"` (394 questions)
- `extract_class` — `class1` (single-hop extraction) / `class2` (comparison) / `class3` (multi-hop reasoning)
- `answer.is_answerable` — `false` for unanswerable questions; expected answer is `"Unanswerable"`
