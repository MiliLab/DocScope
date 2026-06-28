"""Benchmark loader for DocScope format.

Supports two on-disk layouts:

1. DocScope flat list  (benchmark.json):
   [
     {"id": "...", "split": "test|dev",
      "pdf": {"doc_id_str": "urnuuid..."},
      "question": "...",
      "answer": {"answer_text": "...", "is_answerable": true},
      "evidences": [{"local_id": "e1", "page": N, "bbox": [...], "element_type": "..."}, ...],
      "facts": [{"local_id": "f1", "evidence_local_id": "e1", "text_description": "..."}, ...],
      "extract_class": "class1"}, ...
   ]

2. Legacy slim list (backward compat):
   [{"question_id": "...", "pdf_path": "...", "question_en": "...",
     "answer": "...", "gold_pages": [...]}, ...]

Both are normalised into:
    {
      "question_id": str,
      "doc_id": str,           # urnuuid...
      "question_en": str,
      "answer": str,
      "gold_pages": list[int],
      "is_answerable": bool,
      "extract_class": str | None,
      "_raw": dict,
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalise_docscope(entry: dict) -> dict:
    pdf_block = entry.get("pdf") or {}
    answer_block = entry.get("answer") or {}
    evidences = entry.get("evidences") or []
    pages = sorted({
        int(e["page"]) for e in evidences
        if isinstance(e, dict) and isinstance(e.get("page"), (int, float))
    })
    return {
        "question_id": entry.get("id") or "",
        "doc_id": pdf_block.get("doc_id_str", ""),
        "question_en": entry.get("question", ""),
        "answer": answer_block.get("answer_text", ""),
        "gold_pages": pages,
        "is_answerable": bool(answer_block.get("is_answerable", True)),
        "extract_class": entry.get("extract_class"),
        "split": entry.get("split", ""),
        "_raw": entry,
    }


def _normalise_legacy(entry: dict) -> dict:
    return {
        "question_id": entry.get("question_id", ""),
        "doc_id": "",
        "question_en": entry.get("question_en", ""),
        "answer": entry.get("answer", ""),
        "gold_pages": list(entry.get("gold_pages") or []),
        "is_answerable": True,
        "extract_class": None,
        "split": "",
        "_raw": entry,
    }


def load_benchmark(path: Path | str, split: str | None = None) -> list[dict]:
    """Load benchmark.json; optionally filter by split ('test' or 'dev')."""
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(raw, list):
        # detect format by first record
        if raw and "id" in raw[0] and "pdf" in raw[0]:
            items = [_normalise_docscope(e) for e in raw]
        else:
            items = [_normalise_legacy(e) for e in raw]
    elif isinstance(raw, dict) and "data" in raw:
        # legacy {metadata, data} wrapper — kept for compat
        items = [_normalise_docscope(e) for e in raw["data"]]
    else:
        raise ValueError(f"Unsupported benchmark format at {path}")

    if split:
        items = [q for q in items if q["split"] == split]
    return items
