"""Shared plumbing for the inference pipeline.

Consolidates:
  - Prompts and parsing helpers (re-exported from reasoning_prompts).
  - PDF page rendering with on-disk caching.
  - Page-image encoding helpers (data URI / raw base64).
  - JSONL resume / thread-safe append helpers.
  - Logging helper with a single print lock.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from threading import Lock

import fitz  # PyMuPDF

from benchmark_io import load_benchmark

from reasoning_prompts import (
    INFER_SIMPLE_PROMPT,
    INFER_SYSTEM_PROMPT,
    Citation,
    extract_answer,
    extract_citations,
    page_marker_end,
    page_marker_start,
    page_prf,
    parse_model_output,
    validate_bbox,
)

__all__ = [
    # prompts / parsing
    "INFER_SIMPLE_PROMPT",
    "INFER_SYSTEM_PROMPT",
    "Citation",
    "extract_answer",
    "extract_citations",
    "page_marker_end",
    "page_marker_start",
    "page_prf",
    "parse_model_output",
    "validate_bbox",
    # paths / constants
    "PROJECT_ROOT",
    "BENCHMARK_PATH",
    "PDF_DIR",
    "PAGE_CACHE_DIR",
    "RESULTS_DIR",
    "RENDER_DPI",
    "MAX_PAGES",
    "MAX_RETRIES",
    "RETRY_DELAY",
    "RETRY_MAX_WAIT",
    "retry_sleep",
    "CONCURRENCY",
    "HTTP_TIMEOUT",
    "INFER_MAX_TOKENS",
    # helpers
    "log",
    "resolve_pdf_path",
    "render_pdf_pages",
    "page_paths_to_b64",
    "page_paths_to_data_uris",
    "load_completed_ids",
    "append_result",
    "load_benchmark",
]

# ── Paths & shared constants ──────────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
BENCHMARK_PATH  = PROJECT_ROOT / "benchmark.json"
PDF_DIR         = PROJECT_ROOT / "pdfs"
PAGE_CACHE_DIR  = PROJECT_ROOT / "page_cache"
RESULTS_DIR     = PROJECT_ROOT / "results"

RENDER_DPI      = 72
MAX_PAGES       = 100
MAX_RETRIES     = 12
RETRY_DELAY     = 2
RETRY_MAX_WAIT  = 15
CONCURRENCY     = 32
HTTP_TIMEOUT    = 200
INFER_MAX_TOKENS = 4096 * 4


def retry_sleep(attempt: int) -> float:
    return min(RETRY_DELAY * attempt, RETRY_MAX_WAIT)


# ── Logging ───────────────────────────────────────────────────────────────────

_PRINT_LOCK = Lock()


def log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


# ── PDF resolution and page rendering ─────────────────────────────────────────

def resolve_pdf_path(doc_id: str) -> Path | None:
    """Resolve a doc_id (urnuuid...) to an existing PDF file on disk."""
    candidate = PDF_DIR / f"{doc_id}.pdf"
    if candidate.exists():
        return candidate
    return None


def render_pdf_pages(pdf_file: Path, dpi: int = RENDER_DPI) -> list[Path]:
    """Render every page of the PDF to PNG; cache on disk.

    Cached under PAGE_CACHE_DIR/<stem>/<dpi>dpi/.
    Subsequent calls with the same dpi return cached paths without re-rendering.
    """
    cache_dir = PAGE_CACHE_DIR / pdf_file.stem / f"{dpi}dpi"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sentinel = cache_dir / ".done"
    if sentinel.exists():
        return sorted(
            cache_dir.glob("page_*.png"),
            key=lambda p: int(re.search(r"page_(\d+)", p.name).group(1)),
        )

    doc = fitz.open(pdf_file)
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        out_paths: list[Path] = []
        for i, page in enumerate(doc, start=1):
            out = cache_dir / f"page_{i:04d}.png"
            if not out.exists():
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pix.save(out.as_posix())
            out_paths.append(out)
    finally:
        doc.close()
    sentinel.touch()
    return out_paths


def page_paths_to_b64(paths: list[Path]) -> list[str]:
    return [base64.b64encode(p.read_bytes()).decode("ascii") for p in paths]


def page_paths_to_data_uris(paths: list[Path]) -> list[str]:
    uris: list[str] = []
    for p in paths:
        data = p.read_bytes()
        if not data:
            raise RuntimeError(f"Empty page image (possible render race): {p}")
        b64 = base64.b64encode(data).decode("ascii")
        uris.append(f"data:image/png;base64,{b64}")
    return uris


# ── JSONL IO / resume ─────────────────────────────────────────────────────────

_WRITE_LOCK = Lock()


def load_completed_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = r.get("question_id")
        if qid:
            ids.add(qid)
    return ids


def append_result(path: Path, result: dict) -> None:
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
