# Camera-ready metrics

For new models, use `bash scripts/run_eval.sh <model_id>` from `code/`. See [setup and both versioned entry points](../../README.md). `score.py` generates all required judgments; `metrics.py` computes the nine metrics. The prompts are copied from the archived camera-ready evaluation, including its individual prediction extension and citation attribution rules.

- Page/region/fact metrics use micro counts over answerable questions.
- Page precision: unique valid pages / unique predicted pages. Page recall: facts with any cited supporting page / all facts.
- Region precision: individually covered predicted boxes / predicted boxes on hit pages. Region recall: facts with a covered supporting region / facts eligible on hit pages. Exact page/box duplicates count once. Invalid boxes remain in the denominator and cannot hit. Only individual `covered` matches receive credit; box unions do not.
- Fact Coverage: covered facts / eligible facts, with OR across valid cited supporting pages.
- Accuracy: correctly answered questions / all selected questions, including unanswerable questions.
- Complete@Correct: among correctly answered answerable questions, the proportion for which every fact is covered and has an associated citation whose own page and box match that fact's support.

All `evidence_local_ids` links participate. No duplicate-review metadata is required. Fact coverage and citation attribution are separate judgments. Failed or missing judgments prevent a final summary.

## Archived published-table calculation

`calculate.py` retains the original main-table aggregation for historical frozen artifacts:

```bash
python3 eval/camera_ready/calculate.py --root /path/to/artifacts --output /path/to/report
```

It requires the original unmodified `HF-DocScope-v2-20260916/benchmark.json`, `Inference_benchmark_noimg_20260915/results/api_eval`, and the archived `output/` directories: `region_judge_full_32104_20260916`, `citation_attribution_full_qwen38_r10_20260917`, `answer_rejudge_qwen36_20260918`, `fact_judge_alternative_qwen36_20260918`, `complete_resolution_20260918`. Those historical raw artifacts are not bundled. This archival script is separate from the default new-model workflow. `reference_metrics.json` records the published aggregate results.
