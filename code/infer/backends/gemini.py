"""Google Gemini backend (via official REST API)."""

from __future__ import annotations

import json
import os
import time

import requests

from reasoning_common import (
    HTTP_TIMEOUT,
    INFER_MAX_TOKENS,
    INFER_SYSTEM_PROMPT,
    MAX_RETRIES,
    log,
    page_marker_end,
    page_marker_start,
    retry_sleep,
)

from .base import Backend

DEFAULT_BASE_URL    = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_MODEL       = "gemini-2.0-flash"

def _build_parts(question: str, image_b64s: list[str],
                 page_numbers: list[int] | None = None) -> list[dict]:
    parts: list[dict] = []
    for idx, b64 in enumerate(image_b64s, start=1):
        pg = page_numbers[idx - 1] if page_numbers else idx
        parts.append({"text": page_marker_start(pg)})
        parts.append({"inlineData": {"mimeType": "image/png", "data": b64}})
        parts.append({"text": page_marker_end(pg)})
    parts.append({"text": f"Question: {question}"})
    return parts


def _absorb_event(data: dict,
                  text_chunks: list[str],
                  thinking_chunks: list[str]) -> tuple[dict, str]:
    finish_reason = ""
    for cand in data.get("candidates") or []:
        finish_reason = cand.get("finishReason", finish_reason)
        parts = (cand.get("content") or {}).get("parts") or []
        for part in parts:
            if not isinstance(part, dict):
                continue
            try:
                if part.get("thought") is True and "text" in part:
                    thinking_chunks.append(part.get("text") or "")
                    continue
                if "text" in part:
                    text_chunks.append(part.get("text") or "")
            except Exception:
                continue

    um = data.get("usageMetadata") or {}
    usage: dict = {}
    if um:
        usage = {
            "prompt_tokens": um.get("promptTokenCount", 0) or 0,
            "completion_tokens": um.get("candidatesTokenCount", 0) or 0,
            "total_tokens": um.get("totalTokenCount", 0) or 0,
        }
        if "thoughtsTokenCount" in um:
            usage["thoughts_tokens"] = um["thoughtsTokenCount"]
    return usage, finish_reason


def _call(api_key: str, model: str, base_url: str,
          question: str, image_b64s: list[str],
          page_numbers: list[int] | None = None) -> tuple[str, str, dict]:
    url = f"{base_url}/models/{model}:streamGenerateContent?alt=sse"
    payload = {
        "contents": [{"role": "USER", "parts": _build_parts(question, image_b64s, page_numbers)}],
        "systemInstruction": {
            "role": "SYSTEM",
            "parts": [{"text": INFER_SYSTEM_PROMPT}],
        },
        "generationConfig": {
            "maxOutputTokens": INFER_MAX_TOKENS,
            "temperature": 0.0,
            "thinkingConfig": {"thinkingLevel": "LOW"},
        },
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 stream=True, timeout=HTTP_TIMEOUT)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            text_chunks: list[str] = []
            thinking_chunks: list[str] = []
            usage: dict = {}
            finish = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                ev_usage, ev_finish = _absorb_event(event, text_chunks, thinking_chunks)
                if ev_usage:
                    usage = ev_usage
                if ev_finish:
                    finish = ev_finish
            text = "".join(text_chunks)
            thinking = "".join(thinking_chunks)
            if not text and finish and finish.upper() not in {"STOP", "MAX_TOKENS"}:
                raise RuntimeError(f"empty text, finishReason={finish}")
            return text.strip(), thinking, usage
        except Exception as exc:
            last_exc = exc
            log(f"     [gemini retry {attempt}/{MAX_RETRIES}] {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(retry_sleep(attempt))
    raise RuntimeError(f"gemini call failed after {MAX_RETRIES} retries: {last_exc}")


def make_backend(model: str = DEFAULT_MODEL,
                 api_key_env: str = DEFAULT_API_KEY_ENV,
                 base_url: str = DEFAULT_BASE_URL) -> Backend:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var {api_key_env}")

    def call(question: str, images: list[str],
             page_numbers: list[int] | None = None) -> tuple[str, str, dict]:
        return _call(api_key, model, base_url, question, images, page_numbers)

    return Backend(name="gemini", api_key_env=api_key_env,
                   image_format="base64", model=model, call=call)
