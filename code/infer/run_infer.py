#!/usr/bin/env python3
"""Inference entry point for DocScope benchmark.

Usage:
    python infer/run_infer.py \\
        --backend claude \\
        --model claude-opus-4-5 \\
        --output results/run_01.jsonl

    python infer/run_infer.py \\
        --backend openai \\
        --model gpt-4o \\
        --base_url https://api.openai.com/v1 \\
        --output results/run_01.jsonl

    python infer/run_infer.py \\
        --backend gemini \\
        --model gemini-2.0-flash \\
        --output results/run_01.jsonl

Each record written to --output:
    {
      "question_id": str,
      "model": str,
      "model_raw": str,          # full model response text
      "thinking": str,           # extended reasoning trace (if available)
      "parsed": {                # structured parse of model_raw
        "answer": str | null,
        "citations": [...],
        "predicted_pages": [...],
        ...
      },
      "infer_usage": {...},      # token counts
      "status": "ok" | "error: ..."
    }

Resumable: skips question_ids already present in --output.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reasoning_common import (
    BENCHMARK_PATH,
    CONCURRENCY,
    MAX_PAGES,
    RESULTS_DIR,
    append_result,
    load_benchmark,
    load_completed_ids,
    log,
    page_paths_to_b64,
    page_paths_to_data_uris,
    parse_model_output,
    render_pdf_pages,
    resolve_pdf_path,
)


def _build_backend(args):
    if args.backend == "claude":
        from backends.claude import make_backend
        kwargs = dict(model=args.model)
        if args.api_key_env:
            kwargs["api_key_env"] = args.api_key_env
        if args.base_url:
            kwargs["url"] = args.base_url
        return make_backend(**kwargs)

    if args.backend == "openai":
        from backends.openai_compat import make_backend
        kwargs = dict(model=args.model)
        if args.api_key_env:
            kwargs["api_key_env"] = args.api_key_env
        if args.base_url:
            kwargs["base_url"] = args.base_url
        return make_backend(**kwargs)

    if args.backend == "gemini":
        from backends.gemini import make_backend
        kwargs = dict(model=args.model)
        if args.api_key_env:
            kwargs["api_key_env"] = args.api_key_env
        if args.base_url:
            kwargs["base_url"] = args.base_url
        return make_backend(**kwargs)

    raise ValueError(f"Unknown backend: {args.backend}")


def _infer_one(item: dict, backend, dpi: int, max_pages: int) -> dict:
    qid = item["question_id"]
    doc_id = item["doc_id"]
    question = item["question_en"]

    pdf_path = resolve_pdf_path(doc_id)
    if pdf_path is None:
        return {
            "question_id": qid,
            "model": backend.model,
            "status": f"error: PDF not found for {doc_id}",
        }

    try:
        all_pages = render_pdf_pages(pdf_path, dpi=dpi)
    except Exception as e:
        return {
            "question_id": qid,
            "model": backend.model,
            "status": f"error: render failed: {e}",
        }

    pages = all_pages[:max_pages]
    page_numbers = list(range(1, len(pages) + 1))

    try:
        if backend.image_format == "base64":
            images = page_paths_to_b64(pages)
        else:
            images = page_paths_to_data_uris(pages)

        text, thinking, usage = backend.call(question, images, page_numbers)
        parsed = parse_model_output(text)

        return {
            "question_id": qid,
            "model": backend.model,
            "model_raw": text,
            "thinking": thinking,
            "parsed": parsed,
            "infer_usage": usage,
            "status": "ok",
        }
    except Exception as e:
        return {
            "question_id": qid,
            "model": backend.model,
            "status": f"error: {e}",
        }


def main():
    parser = argparse.ArgumentParser(description="DocScope benchmark inference")
    parser.add_argument("--backend", required=True, choices=["claude", "openai", "gemini"])
    parser.add_argument("--model", required=True, help="Model name/ID")
    parser.add_argument("--benchmark", default=str(BENCHMARK_PATH),
                        help="Path to benchmark.json")
    parser.add_argument("--split", choices=["test", "dev"], default=None,
                        help="Filter by split (default: all)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--max_pages", type=int, default=MAX_PAGES)
    parser.add_argument("--dpi", type=int, default=72,
                        help="Render DPI for PDF pages (default: 72)")
    # backend-specific
    parser.add_argument("--base_url", default=None,
                        help="Override API base URL")
    parser.add_argument("--api_key_env", default=None,
                        help="Env var name for API key")
    args = parser.parse_args()

    backend = _build_backend(args)
    log(f"Backend: {backend.name} / model: {backend.model}")

    benchmark = load_benchmark(args.benchmark, split=args.split)
    log(f"Loaded {len(benchmark)} questions from {args.benchmark}"
        + (f" (split={args.split})" if args.split else ""))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_completed_ids(output_path)
    pending = [q for q in benchmark if q["question_id"] not in done]
    log(f"Pending: {len(pending)}  (already done: {len(done)})")

    if not pending:
        log("Nothing to do.")
        return

    ok = err = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_infer_one, item, backend, args.dpi, args.max_pages): item
            for item in pending
        }
        for future in as_completed(futures):
            result = future.result()
            append_result(output_path, result)
            if result.get("status") == "ok":
                ok += 1
                answer = (result.get("parsed") or {}).get("answer") or ""
                log(f"  [ok] {result['question_id']}  ans={answer[:60]!r}")
            else:
                err += 1
                log(f"  [err] {result['question_id']}  {result.get('status')}")

    log(f"\nDone. ok={ok}  err={err}  output={output_path}")


if __name__ == "__main__":
    main()
