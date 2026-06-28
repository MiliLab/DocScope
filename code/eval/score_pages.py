#!/usr/bin/env python3
"""Step 1 — page-level recall/precision (no LLM call).

Reads `infer.jsonl` and computes per-question page set retrieval against
`evidences[*].page` from the benchmark file. Writes `pages.jsonl` (one
row per question) and prints macro / micro aggregates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "infer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reasoning_prompts import page_prf  # noqa: E402

from eval_common import (  # noqa: E402
    DEFAULT_BENCHMARK,
    append_jsonl,
    load_benchmark_index,
    load_completed_keys,
    load_jsonl,
    log,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infer", type=Path, required=True,
                   help="infer.jsonl produced by infer/run_infer.py")
    p.add_argument("--output", type=Path, required=True,
                   help="output pages.jsonl")
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    bench = load_benchmark_index(args.benchmark)
    log(f"Loaded {len(bench)} benchmark entries")

    records = load_jsonl(args.infer)
    log(f"Loaded {len(records)} infer records from {args.infer}")

    done = load_completed_keys(args.output)
    pending = [r for r in records if r.get("question_id") not in done]
    log(f"Already scored: {len(done)} | pending: {len(pending)}")

    sum_p = sum_r = sum_f1 = 0.0
    n = 0
    tp = fp = fn = 0
    for r in pending:
        qid = r.get("question_id")
        if r.get("status") != "ok":
            append_jsonl(args.output, {
                "question_id": qid,
                "status": r.get("status"),
                "skipped": True,
            })
            continue
        parsed = r.get("parsed") or {}
        pred_pages = parsed.get("predicted_pages") or []
        be = bench.get(str(qid))
        gold_pages = be.gold_pages() if be else (r.get("gold_pages") or [])
        prf = page_prf(pred_pages, gold_pages)
        out = {
            "question_id": qid,
            "gold_pages": prf["gold"],
            "predicted_pages": prf["predicted"],
            "tp": prf["tp"], "fp": prf["fp"], "fn": prf["fn"],
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "status": "ok",
        }
        append_jsonl(args.output, out)
        sum_p += prf["precision"]; sum_r += prf["recall"]; sum_f1 += prf["f1"]
        tp += prf["tp"]; fp += prf["fp"]; fn += prf["fn"]
        n += 1

    if n:
        log("─" * 60)
        log(f"Macro over {n}: P={sum_p/n:.3f} R={sum_r/n:.3f} F1={sum_f1/n:.3f}")
        micro_p = tp / (tp + fp) if (tp + fp) else 0.0
        micro_r = tp / (tp + fn) if (tp + fn) else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                    if (micro_p + micro_r) else 0.0)
        log(f"Micro: P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f} "
            f"(tp={tp} fp={fp} fn={fn})")
    log(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
