#!/usr/bin/env python3
"""Step 3 — fact consistency judge (OpenAI-compatible text model).

For every (question, page) where the page is in BOTH `evidences[*].page`
and `parsed.predicted_pages`, build a per-page batch of facts (one row
per `evidence_local_id`-linked fact), call the text judge with the
prompt at `prompts/factucal_consistency.txt`, and write one row per
(question_id, page, fact_id) into `facts.jsonl`.
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
    load_completed_compound_keys,
    load_jsonl,
    log,
    make_judge_client,
    parse_judge_json,
    stream_chat,
)

PROMPT_PATH = PROJECT_ROOT / "prompts" / "factucal_consistency.txt"


def _format_facts_block(facts_on_page: list[dict]) -> tuple[str, list[dict]]:
    lines: list[str] = []
    meta: list[dict] = []
    for fa in facts_on_page:
        fid = fa.get("local_id") or f"f{len(meta)+1}"
        text = fa.get("text_description", "") or ""
        lines.append(f"id={fid} :: {text}")
        meta.append({
            "fact_id": fid,
            "evidence_local_id": fa.get("evidence_local_id"),
            "key_entity": fa.get("key_entity"),
            "key_value": fa.get("key_value"),
            "fact_text": text,
        })
    return "\n".join(lines), meta


def _format_siblings(facts_off_page: list[dict]) -> str:
    if not facts_off_page:
        return "(none)"
    return "; ".join(f.get("text_description", "") or "" for f in facts_off_page)


def process_page(client, model: str, template: str, output_path: Path,
                 qid: str, page: int, bench_entry, record: dict) -> None:
    facts_on = bench_entry.facts_on_page(page)
    if not facts_on:
        return
    facts_off = bench_entry.facts_off_page(page)
    facts_block, meta = _format_facts_block(facts_on)
    sibling_facts = _format_siblings(facts_off)
    user_text = template.format(
        question=bench_entry.question,
        gt_page=page,
        gold_facts_block=facts_block,
        sibling_facts=sibling_facts,
        model_raw=record.get("model_raw", ""),
    )

    try:
        text, usage = stream_chat(
            client, model,
            messages=[{"role": "user", "content": user_text}],
            tag="fact-judge",
        )
    except Exception as exc:
        for m in meta:
            append_jsonl(output_path, {
                "question_id": qid, "page": page,
                **m,
                "label": None, "reason": str(exc),
                "status": "judge_error",
            })
        return

    parsed = parse_judge_json(text) or {}
    items = parsed.get("items") or []
    by_id = {it.get("id"): it for it in items if isinstance(it, dict)}
    for m in meta:
        it = by_id.get(m["fact_id"]) or {}
        append_jsonl(output_path, {
            "question_id": qid, "page": page,
            **m,
            "label": it.get("label"),
            "reason": it.get("reason"),
            "judge_raw": text,
            "judge_usage": usage,
            "status": "ok" if it.get("label") in {"consistent", "not_consistent"}
                       else "judge_parse_error",
        })


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
    log(f"Loaded {len(records)} infer records from {args.infer}")

    done = load_completed_compound_keys(args.output, ("question_id", "page", "fact_id"))
    log(f"Already scored (qid,page,fact) keys: {len(done)}")

    work: list[tuple[str, int, dict]] = []
    for r in records:
        if r.get("status") != "ok":
            continue
        qid = str(r.get("question_id"))
        be = bench.get(qid)
        if be is None:
            continue
        gold_pages = set(be.gold_pages())
        pred_pages = set((r.get("parsed") or {}).get("predicted_pages") or [])
        for page in sorted(gold_pages & pred_pages):
            facts = be.facts_on_page(page)
            if not facts:
                continue
            page_keys = {(qid, page, f.get("local_id")) for f in facts
                         if f.get("local_id")}
            if page_keys and page_keys.issubset(done):
                continue
            work.append((qid, page, r))
    if args.limit > 0:
        work = work[: args.limit]
    log(f"Pending pages this run: {len(work)}")

    if not work:
        log("Nothing to do.")
        return 0

    client = make_judge_client()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(process_page, client, args.judge_model, template,
                        args.output, qid, page, bench[qid], rec): (qid, page)
            for qid, page, rec in work
        }
        for fut in as_completed(futures):
            qp = futures[fut]
            try:
                fut.result()
                log(f"[fact ok] {qp[0]} p{qp[1]}")
            except Exception as exc:
                log(f"[fact fatal] {qp[0]} p{qp[1]}: {exc}")
    log(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
