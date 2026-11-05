# Camera-ready metrics

For new models, use `bash scripts/run_eval.sh <model_id>` from `code/`. See [setup and both versioned entry points](../../README.md). `score.py` generates all required judgments; `metrics.py` computes the nine metrics. The prompts implement individual predicted-box matching, fact coverage, citation attribution, and answer verification.

- Page/region/fact metrics use micro counts over answerable questions.
- Page precision: unique valid pages / unique predicted pages. Page recall: facts with any cited supporting page / all facts.
- Region precision: individually covered predicted boxes / predicted boxes on hit pages. Region recall: facts with a covered supporting region / facts eligible on hit pages. Exact page/box duplicates count once. Invalid boxes remain in the denominator and cannot hit. Only individual `covered` matches receive credit; box unions do not.
- Fact Coverage: covered facts / eligible facts, with OR across valid cited supporting pages.
- Accuracy: correctly answered questions / all selected questions, including unanswerable questions.
- Complete@Correct: among correctly answered answerable questions, the proportion for which every fact is covered and has an associated citation whose own page and box match that fact's support.

All `evidence_local_ids` links participate. No duplicate-review metadata is required. Fact coverage and citation attribution are separate judgments. Failed or missing judgments prevent a final summary.
