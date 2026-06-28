"""Anthropic Claude backend."""

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

DEFAULT_URL         = "https://api.anthropic.com/v1/messages"
DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_MODEL       = "claude-opus-4-7"


def _consume_stream(resp: requests.Response) -> tuple[str, str, dict]:
    text = ""
    thinking = ""
    input_tokens = 0
    output_tokens = 0
    block_kinds: dict[int, str] = {}

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

        etype = event.get("type")
        if etype == "message_start":
            usage = event.get("message", {}).get("usage", {}) or {}
            input_tokens = usage.get("input_tokens", input_tokens) or input_tokens
        elif etype == "content_block_start":
            try:
                idx = event.get("index", 0)
                block = event.get("content_block") or {}
                block_kinds[idx] = block.get("type", "text")
                if block.get("type") == "thinking":
                    thinking += block.get("thinking", "") or ""
            except Exception:
                pass
        elif etype == "content_block_delta":
            delta = event.get("delta", {}) or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                text += delta.get("text", "")
            elif dtype == "thinking_delta":
                try:
                    thinking += delta.get("thinking", "") or ""
                except Exception:
                    pass
        elif etype == "message_delta":
            usage = event.get("usage", {}) or {}
            if "output_tokens" in usage:
                output_tokens = usage["output_tokens"]
            if "input_tokens" in usage:
                input_tokens = usage["input_tokens"]
        elif etype == "message_stop":
            break

    return text, thinking, {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": (input_tokens or 0) + (output_tokens or 0),
    }


def _call(api_key: str, model: str, url: str,
          question: str, image_b64s: list[str],
          page_numbers: list[int] | None = None) -> tuple[str, str, dict]:
    content_blocks: list[dict] = []
    for idx, b64 in enumerate(image_b64s, start=1):
        pg = page_numbers[idx - 1] if page_numbers else idx
        content_blocks.append({"type": "text", "text": page_marker_start(pg)})
        content_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })
        content_blocks.append({"type": "text", "text": page_marker_end(pg)})
    content_blocks.append({"type": "text", "text": f"Question: {question}"})

    payload = {
        "model": model,
        "max_tokens": INFER_MAX_TOKENS,
        "stream": True,
        "system": INFER_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content_blocks}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 stream=True, timeout=HTTP_TIMEOUT)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            text, thinking, usage = _consume_stream(resp)
            return text.strip(), thinking, usage
        except Exception as exc:
            last_exc = exc
            log(f"     [claude retry {attempt}/{MAX_RETRIES}] {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(retry_sleep(attempt))
    raise RuntimeError(f"claude call failed after {MAX_RETRIES} retries: {last_exc}")


def make_backend(model: str = DEFAULT_MODEL,
                 api_key_env: str = DEFAULT_API_KEY_ENV,
                 url: str = DEFAULT_URL) -> Backend:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var {api_key_env}")

    def call(question: str, images: list[str],
             page_numbers: list[int] | None = None) -> tuple[str, str, dict]:
        return _call(api_key, model, url, question, images, page_numbers)

    return Backend(name="claude", api_key_env=api_key_env,
                   image_format="base64", model=model, call=call)
