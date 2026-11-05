# Camera-ready evaluation

The released benchmark contains 1,123 questions (729 test, 394 dev), with 847 additional evidence regions. Download `benchmark.json` from https://huggingface.co/datasets/MiliLab/DocScope.

## Full aggregation from frozen judge artifacts

`calculate.py` is the original main-table aggregation implementation, with configurable root/output paths and relocation of frozen result paths:

```bash
python3 eval/camera_ready/calculate.py --root /path/to/artifacts --output /path/to/new-report
```

The root must contain `HF-DocScope-v2-20260916/benchmark.json`, `Inference_benchmark_noimg_20260915/results/api_eval`, and these subdirectories of `output/`: `region_judge_full_32104_20260916`, `citation_attribution_full_qwen38_r10_20260917`, `answer_rejudge_qwen36_20260918`, `fact_judge_alternative_qwen36_20260918`, `complete_resolution_20260918`. Raw generations, page images and full judge artifacts are not bundled in this code repository. The script validates source hashes, reference alignment, prediction-level matches and missing-judgment bounds before accepting final metrics.

## Metric definitions

- All page, region and fact metrics use micro counts over answerable questions.
- Page precision counts unique valid pages / unique predicted pages. Page recall counts facts with any cited supporting page / all reference facts.
- Region precision counts individually covered predicted boxes / predicted boxes on hit pages. Region recall counts facts with any individually covered supporting region / facts eligible on hit pages. Identical page/box predictions are deduplicated. Invalid boxes retain their denominator and cannot hit. Box unions receive no credit.
- Fact Coverage ORs consistency judgments over each fact's valid cited support pages, counting each fact once.
- Accuracy includes all 729 test questions.
- Complete@Correct considers correctly answered answerable questions. Every reference fact must be covered and associated with at least one citation whose own valid page and box cover that fact's support. Coverage and citation association are separate judgments.
- `evidence_local_ids` lists alternative supports; legacy `evidence_local_id` is the fallback. Both accepted and partial supplementary links are used in this evaluation.

The existing `scripts/run_eval.sh`, `score_pages.py`, `score_bbox.py` and `scripts/summarize_eval.py` retain the original diagnostic pipeline. Their legacy page-set recall and region-label summaries are not the camera-ready table metrics. For new model evaluation, prediction-level region matches and citation attribution are required before applying the full aggregation logic; the legacy pipeline alone does not produce those artifacts. The shared fact loader now recognizes all alternative evidence links.
