#!/usr/bin/env python3
"""Step 4 — answer consistency judge (OpenAI-compatible text model).

For each question:
  - `model_answer` = `parsed.answer` if present, else the entire `model_raw`.
  - `gold_answer`  = `"Unanswerable"` when the benchmark marks the question
                     unanswerable; otherwise the original gold answer text.
  - prompt: `prompts/answer_verification.txt`.

Output (`answer.jsonl`): one row per question.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "infer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (  # noqa: E402
    _judge_model,
    DEFAULT_BENCHMARK,
    EVAL_CONCURRENCY,
    append_jsonl,
    load_benchmark_index,
    load_completed_keys,
    load_jsonl,
    log,
    make_judge_client,
    parse_judge_json,
    stream_chat,
)

PROMPT_PATH = PROJECT_ROOT / "prompts" / "answer_verification.txt"


def process_record(client, model: str, template: str, output_path: Path,
                   record: dict, bench_entry) -> None:
    qid = str(record.get("question_id"))
    parsed = record.get("parsed") or {}
    has_answer_tag = bool(parsed.get("has_answer_tag"))
    if has_answer_tag and parsed.get("answer"):
        model_answer = parsed.get("answer", "")
    else:
        model_answer = record.get("model_raw", "")

    if bench_entry is not None and not bench_entry.is_answerable:
        gold_answer = "Unanswerable"
    else:
        gold_answer = (bench_entry.answer_text if bench_entry
                       else record.get("gold_answer", ""))

    user_text = template.format(
        question=(bench_entry.question if bench_entry
                  else record.get("question", "")),
        gold_answer=gold_answer,
        model_answer=model_answer,
    )

    try:
        text, usage = stream_chat(
            client, model,
            messages=[{"role": "user", "content": user_text}],
            tag="answer-judge",
        )
    except Exception as exc:
        append_jsonl(output_path, {
            "question_id": qid,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            "has_answer_tag": has_answer_tag,
            "consistent": None,
            "reason": str(exc),
            "status": "judge_error",
        })
        return

    parsed_judge = parse_judge_json(text) or {}
    consistent = parsed_judge.get("consistent")
    reason = parsed_judge.get("reason") or parsed_judge.get("reasoning") or ""
    out = {
        "question_id": qid,
        "gold_answer": gold_answer,
        "model_answer": model_answer,
        "has_answer_tag": has_answer_tag,
        "consistent": (bool(consistent) if isinstance(consistent, bool)
                       else None),
        "reason": reason,
        "judge_raw": text,
        "judge_usage": usage,
        "status": "ok" if isinstance(consistent, bool) else "judge_parse_error",
    }
    append_jsonl(output_path, out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    p.add_argument("--judge-model", type=str, default=_judge_model())
    p.add_argument("--concurrency", type=int, default=EVAL_CONCURRENCY)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    template = PROMPT_PATH.read_text(encoding="utf-8")
    bench = load_benchmark_index(args.benchmark)
    log(f"Loaded {len(bench)} benchmark entries")

    records = load_jsonl(args.infer)
    log(f"Loaded {len(records)} infer records")

    done = load_completed_keys(args.output)
    pending = [r for r in records if str(r.get("question_id")) not in done
               and r.get("status") == "ok"]
    if args.limit > 0:
        pending = pending[: args.limit]
    log(f"Already scored: {len(done)} | pending: {len(pending)}")

    if not pending:
        log("Nothing to do.")
        return 0

    client = make_judge_client()

    def _wrapped(rec):
        be = bench.get(str(rec.get("question_id")))
        process_record(client, args.judge_model, template, args.output, rec, be)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_wrapped, r): r.get("question_id") for r in pending}
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                fut.result()
                log(f"[answer ok] {qid}")
            except Exception as exc:
                log(f"[answer fatal] {qid}: {exc}")

    # Aggregate
    rows = load_jsonl(args.output)
    judged = [r for r in rows if isinstance(r.get("consistent"), bool)]
    if judged:
        acc = sum(1 for r in judged if r["consistent"]) / len(judged)
        log("─" * 60)
        log(f"Answer accuracy: {acc:.4f} ({len(judged)} judged)")
    log(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
