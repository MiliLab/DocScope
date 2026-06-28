#!/usr/bin/env python3
"""Step 2 — BBox grounding judge (OpenAI-compatible multimodal model).

For every (question, page) where the page is in BOTH:
  - the benchmark `evidences[*].page` set (gold pages), and
  - the model's `parsed.predicted_pages` set (pred pages),

render a 144 DPI page with all GOLD[i] (green, absolute pixels) and
PRED[j] boxes (red, normalized → multiply by image W/H). Pred citations
whose bbox is not normalized are NOT painted (recorded as
`skipped_pred_bboxes`).

The image + the prompt at `prompts/evidence_grounding.txt` are sent
to the multimodal judge; we expect a JSON `{"items": [{"id","label","reason"},...]}`
with one entry per `gold` evidence on that page (id `g{i}` 1-indexed).

Output (`bbox.jsonl`): one row per (question_id, page, gold_id).
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
    _bbox_judge_model,
    DEFAULT_BENCHMARK,
    EVAL_CONCURRENCY,
    append_jsonl,
    load_benchmark_index,
    load_completed_compound_keys,
    load_jsonl,
    log,
    make_bbox_judge_client,
    parse_judge_json,
    png_to_data_uri,
    stream_chat,
)
from render_eval_page import render_pair  # noqa: E402

PROMPT_PATH = PROJECT_ROOT / "prompts" / "evidence_grounding.txt"


def _format_gold_block(evs_on_page: list[dict]) -> tuple[str, list[dict]]:
    lines: list[str] = []
    g_meta: list[dict] = []
    for i, ev in enumerate(evs_on_page, start=1):
        gid = f"g{i}"
        bbox = ev.get("bbox") or []
        line = (
            f"id={gid} :: index_on_page={i}, "
            f"element_type={ev.get('element_type', 'unknown')}, "
            f"gold_bbox_px={bbox}"
        )
        lines.append(line)
        g_meta.append({
            "gold_id": gid,
            "evidence_local_id": ev.get("local_id"),
            "element_type": ev.get("element_type"),
            "gold_bbox_px": bbox,
        })
    return "\n".join(lines), g_meta


def build_prompt(template: str, question: str, gold_answer: str,
                 gt_page: int, gold_total_on_page: int,
                 gold_facts_block: str,
                 n_pred: int, pred_bboxes_norm: list[list[float]]) -> str:
    return template.format(
        question=question,
        gold_answer=gold_answer,
        gt_page=gt_page,
        gold_total_on_page=gold_total_on_page,
        gold_facts_block=gold_facts_block,
        n_pred=n_pred,
        pred_bboxes_norm=pred_bboxes_norm,
    )


def process_page(client, model: str, template: str, output_path: Path,
                 imgs_dir: Path, qid: str, page: int, bench_entry,
                 record: dict) -> None:
    evs_on_page = bench_entry.evidences_on_page(page)
    if not evs_on_page:
        return

    citations = (record.get("parsed") or {}).get("citations") or []
    pred_on_page = [c for c in citations if c.get("page") == page]
    pred_norm: list[list[float]] = []
    pred_skipped: list[list[float]] = []
    for c in pred_on_page:
        bbox = c.get("bbox") or []
        if len(bbox) != 4:
            continue
        if c.get("is_normalized", False):
            pred_norm.append(bbox)
        else:
            pred_skipped.append(bbox)

    gold_block, g_meta = _format_gold_block(evs_on_page)

    img_path = imgs_dir / f"{qid.replace('::', '__')}__p{page}.png"
    rendered = render_pair(
        bench_entry.doc_id, page,
        [ev.get("bbox") for ev in evs_on_page],
        pred_norm, img_path,
    )
    if not rendered:
        for gm in g_meta:
            append_jsonl(output_path, {
                "question_id": qid, "page": page,
                **gm,
                "label": None, "reason": "render_failed",
                "status": "render_failed",
                "skipped_pred_bboxes": pred_skipped,
            })
        return

    user_text = build_prompt(
        template,
        question=bench_entry.question,
        gold_answer=bench_entry.answer_text,
        gt_page=page,
        gold_total_on_page=len(evs_on_page),
        gold_facts_block=gold_block,
        n_pred=len(pred_norm),
        pred_bboxes_norm=pred_norm,
    )

    try:
        text, usage = stream_chat(
            client, model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": png_to_data_uri(img_path)}},
                    {"type": "text", "text": user_text},
                ],
            }],
            tag="bbox-judge",
            temperature=None,
        )
    except Exception as exc:
        for gm in g_meta:
            append_jsonl(output_path, {
                "question_id": qid, "page": page, **gm,
                "label": None, "reason": str(exc),
                "status": "judge_error",
                "skipped_pred_bboxes": pred_skipped,
            })
        return

    parsed = parse_judge_json(text) or {}
    items = parsed.get("items") or []
    by_id = {it.get("id"): it for it in items if isinstance(it, dict)}
    for gm in g_meta:
        it = by_id.get(gm["gold_id"]) or {}
        append_jsonl(output_path, {
            "question_id": qid,
            "page": page,
            **gm,
            "n_pred_drawn": len(pred_norm),
            "skipped_pred_bboxes": pred_skipped,
            "label": it.get("label"),
            "reason": it.get("reason"),
            "judge_raw": text,
            "judge_usage": usage,
            "status": "ok" if it.get("label") in
                       {"covered", "imprecise", "not_covered"} else "judge_parse_error",
        })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--infer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    p.add_argument("--imgs-dir", type=Path, default=None,
                   help="where to cache rendered pair PNGs (default: <output_parent>/imgs)")
    p.add_argument("--judge-model", type=str, default=_bbox_judge_model())
    p.add_argument("--concurrency", type=int, default=EVAL_CONCURRENCY)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imgs_dir = args.imgs_dir or (args.output.parent / "imgs")
    imgs_dir.mkdir(parents=True, exist_ok=True)

    template = PROMPT_PATH.read_text(encoding="utf-8")
    bench = load_benchmark_index(args.benchmark)
    log(f"Loaded {len(bench)} benchmark entries")

    records = load_jsonl(args.infer)
    log(f"Loaded {len(records)} infer records from {args.infer}")

    done = load_completed_compound_keys(args.output, ("question_id", "page", "gold_id"))
    log(f"Already scored (qid,page,gold) keys: {len(done)}")

    # Build per-page work units up-front so we can dispatch in parallel.
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
        eligible = sorted(gold_pages & pred_pages)
        for page in eligible:
            evs = be.evidences_on_page(page)
            if not evs:
                continue
            # Skip if EVERY (qid,page,gold_id) for this page is already in `done`
            page_keys = {(qid, page, f"g{i}") for i in range(1, len(evs) + 1)}
            if page_keys.issubset(done):
                continue
            work.append((qid, page, r))
    if args.limit > 0:
        work = work[: args.limit]
    log(f"Pending pages this run: {len(work)} (concurrency={args.concurrency})")

    if not work:
        log("Nothing to do.")
        return 0

    client = make_bbox_judge_client()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(process_page, client, args.judge_model, template,
                        args.output, imgs_dir, qid, page, bench[qid], rec): (qid, page)
            for qid, page, rec in work
        }
        for fut in as_completed(futures):
            qp = futures[fut]
            try:
                fut.result()
                log(f"[bbox ok] {qp[0]} p{qp[1]}")
            except Exception as exc:
                log(f"[bbox fatal] {qp[0]} p{qp[1]}: {exc}")
    log(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
