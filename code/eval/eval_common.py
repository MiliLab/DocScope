"""Shared plumbing for the score_*.py evaluation scripts.

Judge configuration is read from environment variables:
  JUDGE_API_KEY   — API key for the judge endpoint
  JUDGE_BASE_URL  — base URL of the judge (OpenAI-compatible)
  JUDGE_MODEL     — model name (default: gpt-4o-mini)

The bbox judge (score_bbox.py) uses the same endpoint as the text judge
but requires a multimodal model.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "infer"))

from reasoning_common import MAX_RETRIES, retry_sleep  # noqa: E402

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BENCHMARK = PROJECT_ROOT / "benchmark.json"

# ── Judge configuration (set via environment variables) ───────────────────────
#
# Text judge  (score_answer, score_facts):
#   JUDGE_API_KEY    — API key
#   JUDGE_BASE_URL   — OpenAI-compatible endpoint
#   JUDGE_MODEL      — model name (default: gpt-4o-mini)
#
# BBox judge  (score_bbox) — must be a multimodal model:
#   BBOX_JUDGE_API_KEY   — API key  (falls back to JUDGE_API_KEY)
#   BBOX_JUDGE_BASE_URL  — endpoint (falls back to JUDGE_BASE_URL)
#   BBOX_JUDGE_MODEL     — model name (default: gpt-4o-mini)

JUDGE_API_KEY_ENV    = "JUDGE_API_KEY"
JUDGE_BASE_URL_ENV   = "JUDGE_BASE_URL"
JUDGE_MODEL_ENV      = "JUDGE_MODEL"
DEFAULT_JUDGE_MODEL  = "gpt-4o-mini"

BBOX_JUDGE_API_KEY_ENV   = "BBOX_JUDGE_API_KEY"
BBOX_JUDGE_BASE_URL_ENV  = "BBOX_JUDGE_BASE_URL"
BBOX_JUDGE_MODEL_ENV     = "BBOX_JUDGE_MODEL"

JUDGE_MAX_TOKENS = 2048
EVAL_CONCURRENCY = 32


def _judge_model() -> str:
    return os.environ.get(JUDGE_MODEL_ENV, DEFAULT_JUDGE_MODEL)


def _bbox_judge_model() -> str:
    return os.environ.get(BBOX_JUDGE_MODEL_ENV, DEFAULT_JUDGE_MODEL)


# ── Logging / IO ──────────────────────────────────────────────────────────────

_PRINT_LOCK = Lock()
_WRITE_LOCK = Lock()


def log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_jsonl(path: Path, rec: dict) -> None:
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_completed_keys(path: Path, key: str = "question_id") -> set[str]:
    out: set[str] = set()
    for r in load_jsonl(path):
        v = r.get(key)
        if v is not None:
            out.add(str(v))
    return out


def load_completed_compound_keys(path: Path, keys: Iterable[str]) -> set[tuple]:
    out: set[tuple] = set()
    for r in load_jsonl(path):
        try:
            out.add(tuple(r[k] for k in keys))
        except KeyError:
            continue
    return out


# ── Benchmark loader ──────────────────────────────────────────────────────────

@dataclass
class BenchEntry:
    qid: str
    question: str
    doc_id: str
    answer_text: str
    is_answerable: bool
    evidences: list[dict]
    facts: list[dict]
    raw: dict

    def gold_pages(self) -> list[int]:
        return sorted({
            int(e["page"]) for e in self.evidences
            if isinstance(e, dict) and isinstance(e.get("page"), (int, float))
        })

    def evidences_on_page(self, page: int) -> list[dict]:
        return [e for e in self.evidences if e.get("page") == page]

    def facts_on_page(self, page: int) -> list[dict]:
        ev_ids = {e["local_id"] for e in self.evidences_on_page(page) if e.get("local_id")}
        return [f for f in self.facts if f.get("evidence_local_id") in ev_ids]

    def facts_off_page(self, page: int) -> list[dict]:
        ev_ids_on = {e["local_id"] for e in self.evidences_on_page(page) if e.get("local_id")}
        return [f for f in self.facts if f.get("evidence_local_id") not in ev_ids_on]


def load_benchmark_index(path: Path | str) -> dict[str, BenchEntry]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    out: dict[str, BenchEntry] = {}
    for r in rows:
        qid = r.get("id") or r.get("question_id")
        if not qid:
            continue
        pdf_block = r.get("pdf") or {}
        doc_id = pdf_block.get("doc_id_str", "")
        answer_block = r.get("answer") or {}
        out[str(qid)] = BenchEntry(
            qid=str(qid),
            question=r.get("question", ""),
            doc_id=doc_id,
            answer_text=answer_block.get("answer_text", "") or r.get("answer", ""),
            is_answerable=bool(answer_block.get("is_answerable", True)),
            evidences=r.get("evidences") or [],
            facts=r.get("facts") or [],
            raw=r,
        )
    return out


# ── Judge client ──────────────────────────────────────────────────────────────

def make_judge_client() -> OpenAI:
    """Text judge client (score_answer, score_facts)."""
    key = os.environ.get(JUDGE_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"Missing env var {JUDGE_API_KEY_ENV}. "
            "Set JUDGE_API_KEY, JUDGE_BASE_URL, and optionally JUDGE_MODEL."
        )
    base_url = os.environ.get(JUDGE_BASE_URL_ENV)
    if not base_url:
        raise RuntimeError(f"Missing env var {JUDGE_BASE_URL_ENV}.")
    return OpenAI(api_key=key, base_url=base_url)


def make_bbox_judge_client() -> OpenAI:
    """BBox judge client (score_bbox) — must support multimodal input.

    Falls back to JUDGE_API_KEY / JUDGE_BASE_URL if bbox-specific vars are unset.
    """
    key = os.environ.get(BBOX_JUDGE_API_KEY_ENV) or os.environ.get(JUDGE_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"Missing env var {BBOX_JUDGE_API_KEY_ENV} (or {JUDGE_API_KEY_ENV}). "
            "Set BBOX_JUDGE_API_KEY, BBOX_JUDGE_BASE_URL, and optionally BBOX_JUDGE_MODEL."
        )
    base_url = os.environ.get(BBOX_JUDGE_BASE_URL_ENV) or os.environ.get(JUDGE_BASE_URL_ENV)
    if not base_url:
        raise RuntimeError(f"Missing env var {BBOX_JUDGE_BASE_URL_ENV} (or {JUDGE_BASE_URL_ENV}).")
    return OpenAI(api_key=key, base_url=base_url)


def stream_chat(client: OpenAI, model: str, messages: list[dict],
                max_tokens: int = JUDGE_MAX_TOKENS,
                tag: str = "judge",
                temperature: float | None = 0.0) -> tuple[str, dict]:
    """Stream chat completion with retry; returns (text, usage)."""
    last_exc: Exception | None = None
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            stream = client.chat.completions.create(**kwargs)
            text = ""
            usage: dict = {}
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        text += delta.content
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }
            return text.strip(), usage
        except Exception as exc:
            last_exc = exc
            log(f"     [{tag} retry {attempt}/{MAX_RETRIES}] {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(retry_sleep(attempt))
    raise RuntimeError(f"{tag} call failed after {MAX_RETRIES} retries: {last_exc}")


# ── JSON parsing helpers ──────────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_judge_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None


# ── Image helpers ─────────────────────────────────────────────────────────────

def png_to_data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None
