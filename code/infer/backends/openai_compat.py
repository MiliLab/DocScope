"""OpenAI-compatible chat-completions backend.

Works with OpenAI, Azure OpenAI, and any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os
import time

from openai import OpenAI

from reasoning_common import (
    INFER_MAX_TOKENS,
    INFER_SYSTEM_PROMPT,
    MAX_RETRIES,
    log,
    page_marker_end,
    page_marker_start,
    retry_sleep,
)

from .base import Backend

DEFAULT_BASE_URL    = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL       = "gpt-4o"


def _call(client: OpenAI, model: str,
          question: str, image_data_uris: list[str],
          system_prompt: str | None = INFER_SYSTEM_PROMPT,
          page_numbers: list[int] | None = None) -> tuple[str, str, dict]:
    user_content: list[dict] = []
    for idx, uri in enumerate(image_data_uris, start=1):
        pg = page_numbers[idx - 1] if page_numbers else idx
        user_content.append({"type": "text", "text": page_marker_start(pg)})
        user_content.append({"type": "image_url", "image_url": {"url": uri}})
        user_content.append({"type": "text", "text": page_marker_end(pg)})
    user_content.append({"type": "text", "text": f"Question: {question}"})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=INFER_MAX_TOKENS,
                stream=True,
                stream_options={"include_usage": True},
            )
            text = ""
            thinking = ""
            usage: dict = {}
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        text += delta.content
                    try:
                        rc = getattr(delta, "reasoning_content", None) if delta else None
                        if rc:
                            thinking += rc
                    except Exception:
                        pass
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }
            return text.strip(), thinking, usage
        except Exception as exc:
            last_exc = exc
            log(f"     [openai retry {attempt}/{MAX_RETRIES}] {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(retry_sleep(attempt))
    raise RuntimeError(f"openai call failed after {MAX_RETRIES} retries: {last_exc}")


def make_backend(model: str = DEFAULT_MODEL,
                 api_key_env: str = DEFAULT_API_KEY_ENV,
                 base_url: str = DEFAULT_BASE_URL,
                 system_prompt: str | None = INFER_SYSTEM_PROMPT) -> Backend:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var {api_key_env}")
    client = OpenAI(api_key=api_key, base_url=base_url)

    def call(question: str, images: list[str],
             page_numbers: list[int] | None = None) -> tuple[str, str, dict]:
        return _call(client, model, question, images, system_prompt, page_numbers)

    return Backend(name="openai", api_key_env=api_key_env,
                   image_format="data_uri", model=model, call=call)
